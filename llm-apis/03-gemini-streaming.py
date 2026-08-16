import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3.5-flash")

response = model.generate_content(
    "What is RAG? Explain in 5 lines.",
    stream=True
)

for chunk in response:
    print(chunk.text, end="", flush=True)
