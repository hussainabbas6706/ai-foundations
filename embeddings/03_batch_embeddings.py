from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

# 1. Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

client = genai.Client()

# A batch of documents across two distinct topics
documents = [
    # Topic A: Authentication & Security
    "How do I reset my password?",
    "Steps to recover a forgotten login code.",
    "Updating two-factor authentication settings.",
    
    # Topic B: Astronomy & Space
    "How far is Mars from Earth?",
    "The orbit and atmosphere of Jupiter.",
    "Galaxies, stars, and cosmic dust."
]

print(f"Sending a batch of {len(documents)} documents to Gemini...")

# 2. Batch request to gemini-embedding-001
response = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=documents,
    config=types.EmbedContentConfig(
        task_type="SEMANTIC_SIMILARITY"
    )
)

# Extract 768-dim vectors
vectors = [e.values for e in response.embeddings]

# 3. Compute full 6x6 pairwise cosine similarity matrix
sim_matrix = cosine_similarity(vectors)

# Calculate average similarity WITHIN Topic A vs BETWEEN Topic A & B
intra_topic_sim = np.mean([sim_matrix[0][1], sim_matrix[0][2], sim_matrix[1][2]])
inter_topic_sim = np.mean([sim_matrix[0][3], sim_matrix[0][4], sim_matrix[0][5]])

print("\n--- Batch Embedding Analysis ---")
print(f"Total Vectors Generated:   {len(vectors)}")
print(f"Vector Dimension Size:     {len(vectors[0])}")
print(f"Avg Similarity (Same Topic):  {intra_topic_sim:.4f}")
print(f"Avg Similarity (Diff Topic):  {inter_topic_sim:.4f}")