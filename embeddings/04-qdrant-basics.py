import os
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# Initialize clients
gemini_client = genai.Client()
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

def get_embedding(text):
    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values

# --- STEP 1: CREATE A COLLECTION ---
collection_name = "documents"

# Replaced recreate_collection with explicit check and create
if qdrant_client.collection_exists(collection_name):
    qdrant_client.delete_collection(collection_name)

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=3072,         # Gemini embedding dimensions
        distance=Distance.COSINE
    )
)
print(f"Collection '{collection_name}' created")

# --- STEP 2: INDEX DOCUMENTS ---
documents = [
    "Python is a programming language used for AI development",
    "FastAPI is a modern web framework for building APIs",
    "Karachi is the largest city in Pakistan",
    "Lahore is known for its food and culture",
    "RAG stands for Retrieval Augmented Generation",
    "LangChain is a framework for building LLM applications",
    "Embeddings convert text into numerical vectors",
    "Qdrant is a vector database for similarity search",
    "LangGraph is used for building agentic AI systems",
    "Pakistan has a growing tech industry",
]

print("\nIndexing documents...")
points = []
for i, doc in enumerate(documents):
    embedding = get_embedding(doc)
    points.append(PointStruct(
        id=i,
        vector=embedding,
        payload={"text": doc}
    ))

qdrant_client.upsert(
    collection_name=collection_name,
    points=points
)
print(f"{len(documents)} documents indexed")

# --- STEP 3: SEARCH ---
query = "What is used for building AI agents?"
query_embedding = get_embedding(query)

# Replaced search() with modern query_points()
results = qdrant_client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    limit=3
)

print(f"\nQuery: '{query}'")
print("\nTop 3 results:")
for result in results.points:
    print(f"Score: {result.score:.4f} — {result.payload['text']}")