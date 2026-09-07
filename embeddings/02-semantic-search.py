import os
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()
# Use 'text-embedding-004' or 'models/text-embedding-004'
EMBEDDING_MODEL = "gemini-embedding-001"

# 1. Document Corpus
documents = [
    "Biryani is a famous rice dish deeply rooted in Karachi culinary culture.",
    "Lahore is known for its rich history, vibrant food street, and Mughal architecture.",
    "Vector databases store high-dimensional embeddings for fast similarity search.",
    "Gemini models handle multimodal tasks across text, code, and vision.",
    "Python scripts allow fast prototyping for machine learning workflows."
]

def get_embedding(text: str) -> np.ndarray:
    """Generate embedding vector for a single text input."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return np.array(response.embeddings[0].values)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_search(query: str, doc_texts: list[str], doc_embeddings: list[np.ndarray], top_k: int = 2):
    """Find top matching documents for a query."""
    query_vector = get_embedding(query)

    scores = []
    for idx, doc_vector in enumerate(doc_embeddings):
        sim = cosine_similarity(query_vector, doc_vector)
        scores.append((doc_texts[idx], sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

if __name__ == "__main__":
    print("Embedding documents...")
    # Loop over documents using single-text embed requests
    doc_vectors = [get_embedding(doc) for doc in documents]

    query = "Where can I find famous local Pakistani dishes?"
    print(f"\nQuery: '{query}'")
    print("=" * 45)

    results = semantic_search(query, documents, doc_vectors, top_k=2)

    for rank, (doc_text, score) in enumerate(results, start=1):
        print(f"Rank {rank} (Score: {score:.4f}):")
        print(f"  {doc_text}\n")