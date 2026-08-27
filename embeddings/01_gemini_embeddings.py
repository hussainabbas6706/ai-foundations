from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types

# 1. Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 2. Initialize Gemini client
client = genai.Client()

sentences = [
    "How do I reset my password?",
    "Steps to recover a forgotten login code.",
    "What is the capital of France?"
]

print("Generating embeddings with Gemini...")

# 3. Call embed_content using the active gemini-embedding-001 model
response = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=sentences,
    config=types.EmbedContentConfig(
        task_type="SEMANTIC_SIMILARITY"
    )
)

# 4. Extract vectors and calculate pairwise similarity matrix
vectors = [e.values for e in response.embeddings]
sim_matrix = cosine_similarity(vectors)

print("\n--- Similarity Scores ---")
print(f"Sentence 1 vs Sentence 2 (Similar):    {sim_matrix[0][1]:.4f}")
print(f"Sentence 1 vs Sentence 3 (Unrelated):  {sim_matrix[0][2]:.4f}")