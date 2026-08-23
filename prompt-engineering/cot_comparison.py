import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def run_comparison():
    question = "A shop sells 3 items for PKR 150 each. Customer pays PKR 500. What is the change?"

    client = genai.Client()

    # 1. Standard Prompt (Without CoT)
    print("=== WITHOUT CHAIN OF THOUGHT ===")
    response_raw = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=question,
    )
    print(response_raw.text)

    # 2. Chain of Thought Prompt (With Step-by-Step Instructions)
    print("\n=== WITH CHAIN OF THOUGHT ===")
    cot_prompt = f"""{question}

Think step by step:
1. First calculate total cost
2. Then calculate change
3. Give final answer"""

    response_cot = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=cot_prompt,
    )
    print(response_cot.text)

if __name__ == "__main__":
    run_comparison()