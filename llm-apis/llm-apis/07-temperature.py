import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

question = "Give me a creative name for an AI startup."

# Low temperature — deterministic, focused
model_low = genai.GenerativeModel(
    "gemini-3.5-flash",
    generation_config={"temperature": 0.1}
)

# High temperature — creative, random
model_high = genai.GenerativeModel(
    "gemini-3.5-flash", 
    generation_config={"temperature": 1.5}
)

print("=== LOW TEMPERATURE (0.1) — focused ===")
print(model_low.generate_content(question).text)

print("\n=== HIGH TEMPERATURE (1.5) — creative ===")
print(model_high.generate_content(question).text)