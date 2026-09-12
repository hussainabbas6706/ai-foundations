import os
import time
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

# 20 Sentences covering food, locations, weather, and technology
sentences = [
    # Food & Cuisine
    "Biryani is a famous spice-filled rice dish from South Asia.",
    "Karachi is famous for its spicy and flavor-rich biryani.",
    "I love eating hot pizza on a Friday night.",
    "Italian pasta is typically served with tomato or cream sauce.",
    "Traditional Pakistani food relies heavily on aromatic spices.",
    
    # Cities & Travel
    "Karachi is a busy port city located in southern Pakistan.",
    "Lahore is celebrated for its historic culture and lively food streets.",
    "Tokyo is the bustling capital city of Japan.",
    "Traveling to historic cities gives great insight into local culture.",
    "Public transport makes navigating major cities much easier.",
    
    # Technology & AI
    "Artificial intelligence models process natural language effectively.",
    "Machine learning algorithms require large amounts of quality data.",
    "Python is the most popular language for building AI applications.",
    "Modern software development relies heavily on cloud technology.",
    "Smartphones have changed how people communicate every day.",
    
    # Weather & Nature
    "The heavy rain caused flooding across the low-lying areas.",
    "Dark clouds usually indicate an incoming thunderstorm.",
    "Sunny days in the summer are great for outdoor activities.",
    "The forest was peaceful with birds singing in the high trees.",
    "Extreme climate shifts are impacting weather patterns worldwide."
]

print(f"Embedding {len(sentences)} sentences...\n")

# Generate embeddings
embeddings = {}
for sentence in sentences:
    embeddings[sentence] = get_embedding(sentence)
    time.sleep(0.1)  # Brief delay to prevent rate-limit bursts

print(f"Embedding dimension size: {len(list(embeddings.values())[0])}\n")

# Sentence pairs to evaluate similarity
pairs = [
    # High similarity (Same topic)
    (
        "Biryani is a famous spice-filled rice dish from South Asia.",
        "Karachi is famous for its spicy and flavor-rich biryani."
    ),
    (
        "Karachi is a busy port city located in southern Pakistan.",
        "Lahore is celebrated for its historic culture and lively food streets."
    ),
    (
        "Artificial intelligence models process natural language effectively.",
        "Machine learning algorithms require large amounts of quality data."
    ),
    (
        "The heavy rain caused flooding across the low-lying areas.",
        "Dark clouds usually indicate an incoming thunderstorm."
    ),
    
    # Low similarity (Cross topic)
    (
        "I love eating hot pizza on a Friday night.",
        "Python is the most popular language for building AI applications."
    ),
    (
        "Tokyo is the bustling capital city of Japan.",
        "The forest was peaceful with birds singing in the high trees."
    )
]

print("=== SIMILARITY SCORES ===")
for s1, s2 in pairs:
    score = cosine_similarity(embeddings[s1], embeddings[s2])
    print(f"Sentence 1: \"{s1}\"")
    print(f"Sentence 2: \"{s2}\"")
    print(f"Cosine Similarity: {score:.4f}\n")