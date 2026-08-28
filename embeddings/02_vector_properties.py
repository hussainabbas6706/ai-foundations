from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from google import genai

# Load environment variables from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

client = genai.Client()

text = "Understanding vector embeddings in AI systems."

print("Generating single text embedding...")
response = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=text,
)

# Extract vector values
vector = response.embeddings[0].values

# Key properties of vector embeddings
dimension = len(vector)
vector_slice = [round(val, 4) for val in vector[:5]]
magnitude = np.linalg.norm(vector)

print("\n--- Vector Properties ---")
print(f"Input Text:       '{text}'")
print(f"Vector Dimension: {dimension}")
print(f"Sample Dimensions: {vector_slice} ...")
print(f"Vector Magnitude: {magnitude:.4f}")