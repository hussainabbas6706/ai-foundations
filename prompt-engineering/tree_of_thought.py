# prompt-engineering/tree_of_thought.py
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

class CandidateStrategy(BaseModel):
    strategy_title: str
    explanation: str
    feasibility_score: int = Field(description="Score from 1 to 10 on feasibility.")

class EvaluatedStrategies(BaseModel):
    candidates: list[CandidateStrategy]

def tree_of_thought_solve(problem: str) -> str:
    client = genai.Client()

    # Step 1: Branching - Generate 3 distinct initial strategies
    print("Step 1: Generating multiple candidate thoughts (branches)...")
    branch_prompt = f"Problem: {problem}\nGenerate 3 completely different high-level strategies to solve this."
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=branch_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvaluatedStrategies,
            temperature=0.7,
        )
    )
    
    candidates = EvaluatedStrategies.model_validate_json(response.text).candidates

    # Step 2: Search & Pruning - Select the highest-scoring branch
    best_candidate = max(candidates, key=lambda c: c.feasibility_score)
    print(f"\nStep 2: Selected Best Branch -> '{best_candidate.strategy_title}' (Score: {best_candidate.feasibility_score}/10)")

    # Step 3: Deep Exploration - Expand the selected branch into a final solution
    print("Step 3: Expanding the selected branch into a final execution plan...\n")
    expansion_prompt = f"Problem: {problem}\nChosen Strategy: {best_candidate.explanation}\nExecute this strategy in detail."
    
    final_response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=expansion_prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )
    
    return final_response.text

if __name__ == "__main__":
    problem = "Design a zero-downtime migration strategy for a 500GB SQL database to PostgreSQL in a high-traffic e-commerce app."
    
    print("=== TREE OF THOUGHTS EXECUTION ===")
    solution = tree_of_thought_solve(problem)
    print("--- Final Plan ---")
    print(solution)