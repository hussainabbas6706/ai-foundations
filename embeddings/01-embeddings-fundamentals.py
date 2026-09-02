import os
from dotenv import load_dotenv
from google import genai
import numpy as np

load_dotenv()

client = genai.Client()

def get_embedding(text):
    response = client.models.embed_content(
       model="gemini-embedding-001",
        contents=text,
    )
    return np.array(response.embeddings[0].values)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Embed words
words = ["king", "queen", "man", "woman", "biryani", "karachi", "lahore"]

print("Embedding words...\n")
embeddings = {word: get_embedding(word) for word in words}
print(f"Each embedding: {len(list(embeddings.values())[0])} dimensions\n")

# Compare pairs
print("=== SIMILARITY SCORES ===")
pairs = [
    ("king", "queen"),
    ("karachi", "lahore"),
    ("king", "biryani"),
    ("man", "woman"),
]

for w1, w2 in pairs:
    score = cosine_similarity(embeddings[w1], embeddings[w2])
    print(f"{w1} vs {w2}: {score:.4f}")