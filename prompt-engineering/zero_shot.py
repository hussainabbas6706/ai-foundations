# prompt-engineering/zero_shot.py
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

def zero_shot_prompt(task: str, text_input: str) -> str:
    """Sends a direct instruction and target input without providing any prior examples."""
    prompt = f"Task: {task}\n\nInput:\n{text_input}\n\nOutput:"
    
    # Client automatically reads GEMINI_API_KEY loaded by dotenv
    client = genai.Client()
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
        )
    )
    return response.text

if __name__ == "__main__":
    task = "Extract key technical skills as a comma-separated list."
    sample_input = "Frontend developer proficient in React, JavaScript, and Tailwind CSS, with experience in FastAPI backends."

    print("Sending Zero-Shot request to Gemini API...")
    result = zero_shot_prompt(task=task, text_input=sample_input)
    print("\n--- Zero-Shot Output ---")
    print(result)
    print("------------------------")