import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

# Define the structured output schema using Pydantic
class CoTResponse(BaseModel):
    thinking_steps: list[str] = Field(description="List of sequential reasoning steps to solve the problem.")
    final_answer: str = Field(description="The final concise answer derived from the reasoning steps.")

def structured_cot_prompt(problem: str) -> CoTResponse:
    """Executes Chain of Thought prompting and enforces a structured Pydantic response."""
    prompt = f"Problem: {problem}\n\nSolve this problem by listing your step-by-step reasoning first, followed by the final answer."

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CoTResponse,
            temperature=0.1,
        )
    )
    
    # Parse JSON string directly into Pydantic model
    return CoTResponse.model_validate_json(response.text)

if __name__ == "__main__":
    problem = (
        "A server cluster has 3 nodes. Node A processes 45 req/s, "
        "Node B processes 30% more than Node A, and Node C processes half of Node B's capacity. "
        "What is the total cluster capacity in req/s?"
    )

    print("Sending Structured Chain-of-Thought request to Gemini API...\n")
    result = structured_cot_prompt(problem=problem)

    print("--- Reasoning Steps ---")
    for idx, step in enumerate(result.thinking_steps, 1):
        print(f"{idx}. {step}")

    print("\n--- Final Answer ---")
    print(result.final_answer)