import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def few_shot_prompt(task: str, examples: list[dict], text_input: str) -> str:
    """Formats a Few-Shot prompt by prepending structured examples before the target input."""
    
    # 1. Format the examples into a clear pattern
    formatted_examples = ""
    for ex in examples:
        formatted_examples += f"Input:\n{ex['input']}\nOutput:\n{ex['output']}\n\n"
    
    # 2. Construct the full prompt payload
    prompt = f"Task: {task}\n\n{formatted_examples}Input:\n{text_input}\nOutput:"
    
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,  # Low temperature for strict adherence to examples
        )
    )
    return response.text

if __name__ == "__main__":
    task = "Convert informal text into a structured JSON log entry."
    
    # Few-shot examples demonstrating the target format
    examples = [
        {
            "input": "User 104 logged out at 10:15am due to inactivity",
            "output": '{"event": "LOGOUT", "user_id": 104, "reason": "inactivity"}'
        },
        {
            "input": "User 88 failed login attempt at 10:18am wrong password",
            "output": '{"event": "AUTH_FAILURE", "user_id": 88, "reason": "wrong_password"}'
        }
    ]
    
    sample_input = "User 402 updated profile picture successfully"

    print("Sending Few-Shot request to Gemini API...")
    result = few_shot_prompt(task=task, examples=examples, text_input=sample_input)
    print("\n--- Few-Shot Output ---")
    print(result)
    print("-----------------------")