import os
import re
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()
EMBEDDING_MODEL = "gemini-embedding-001"

# Sample long text corpus for chunking demonstration
raw_text = """
Artificial Intelligence is transforming modern technology. Machine learning algorithms allow computers to learn from data and improve over time. Deep learning, a subset of machine learning, uses multi-layered neural networks to model complex patterns.

Pakistani cuisine is renowned for its rich flavors and aromatic spices. Dishes like Biryani, Nihari, and Karahi hold a central place in the country's culinary heritage. Street food in cities like Karachi and Lahore offers a wide variety of kebabs, parathas, and sweets.

Vector databases are specialized storage engines designed for high-dimensional vectors. They enable sub-millisecond similarity search across millions of embedding representations. Common indexing techniques include HNSW and IVF to speed up nearest neighbor retrieval.
"""

def get_embedding(text: str) -> np.ndarray:
    """Generate vector embedding for a single text chunk."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return np.array(response.embeddings[0].values)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1. Fixed-Size Chunking
def fixed_size_chunking(text: str, chunk_size: int = 150) -> list[str]:
    """Splits text strictly every N characters."""
    clean_text = text.strip().replace("\n", " ")
    return [clean_text[i:i + chunk_size] for i in range(0, len(clean_text), chunk_size)]

# 2. Sentence Chunking
def sentence_chunking(text: str) -> list[str]:
    """Splits text naturally at sentence boundaries using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip().replace("\n", " "))
    return [s for s in sentences if s]

# 3. Semantic Chunking
def semantic_chunking(text: str, threshold: float = 0.65) -> list[str]:
    """Splits text dynamically when semantic similarity between consecutive sentences drops."""
    sentences = sentence_chunking(text)
    if len(sentences) <= 1:
        return sentences

    embeddings = [get_embedding(s) for s in sentences]
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(len(sentences) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        # If semantic gap is detected (similarity drops below threshold), split
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# 4. Parent-Child Chunking
def parent_child_chunking(text: str, child_size: int = 100) -> list[dict]:
    """
    Creates large parent chunks (e.g., full paragraphs) and splits them 
    into smaller child chunks for precise retrieval.
    """
    parents = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    structured_data = []

    for parent_id, parent_text in enumerate(parents, start=1):
        # Create smaller children from the parent paragraph
        children = [parent_text[i:i + child_size] for i in range(0, len(parent_text), child_size)]
        
        for child_id, child_text in enumerate(children, start=1):
            structured_data.append({
                "child_id": f"P{parent_id}-C{child_id}",
                "child_text": child_text,
                "parent_id": f"P{parent_id}",
                "parent_text": parent_text
            })

    return structured_data

if __name__ == "__main__":
    print("=== 1. FIXED-SIZE CHUNKING ===")
    fixed_chunks = fixed_size_chunking(raw_text, chunk_size=150)
    for idx, chunk in enumerate(fixed_chunks, start=1):
        print(f"Chunk {idx} ({len(chunk)} chars): {chunk}\n")

    print("=== 2. SENTENCE CHUNKING ===")
    sent_chunks = sentence_chunking(raw_text)
    for idx, chunk in enumerate(sent_chunks, start=1):
        print(f"Chunk {idx}: {chunk}\n")

    print("=== 3. SEMANTIC CHUNKING ===")
    print("Calculating embeddings to detect topic shifts...")
    sem_chunks = semantic_chunking(raw_text, threshold=0.65)
    for idx, chunk in enumerate(sem_chunks, start=1):
        print(f"Semantic Chunk {idx}:\n  {chunk}\n")

    print("=== 4. PARENT-CHILD CHUNKING ===")
    pc_data = parent_child_chunking(raw_text, child_size=120)
    for item in pc_data[:3]:  # Print first 3 relations
        print(f"Child ID: {item['child_id']} -> '{item['child_text']}'")
        print(f"Parent Ref: {item['parent_id']} -> '{item['parent_text'][:60]}...'\n")