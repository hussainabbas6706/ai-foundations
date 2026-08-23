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

def run_tot_comparison(problem: str):
    client = genai.Client()

    # 1. DIRECT PROMPT
    print("========================================")
    print("1. DIRECT PROMPT (ZERO-SHOT)")
    print("========================================")
    direct_res = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"Problem: {problem}\nProvide a complete architecture solution.",
    )
    print(direct_res.text)

    # 2. CHAIN OF THOUGHT (CoT)
    print("\n========================================")
    print("2. CHAIN OF THOUGHT (CoT)")
    print("========================================")
    cot_prompt = f"Problem: {problem}\nThink step-by-step to design the architecture solution."
    cot_res = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=cot_prompt,
    )
    print(cot_res.text)

    # 3. TREE OF THOUGHTS (ToT)
    print("\n========================================")
    print("3. TREE OF THOUGHTS (ToT - Branching & Evaluation)")
    print("========================================")
    
    # Branching
    branch_prompt = f"Problem: {problem}\nGenerate 3 distinct architectural strategies."
    branch_res = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=branch_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvaluatedStrategies,
            temperature=0.7,
        )
    )
    candidates = EvaluatedStrategies.model_validate_json(branch_res.text).candidates
    best_branch = max(candidates, key=lambda c: c.feasibility_score)
    print(f"-> Evaluated {len(candidates)} branches.")
    print(f"-> Selected Winner: '{best_branch.strategy_title}' (Score: {best_branch.feasibility_score}/10)\n")

    # Deep expansion of winning branch
    tot_res = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"Problem: {problem}\nExecute Strategy: {best_branch.explanation}",
    )
    print(tot_res.text)

if __name__ == "__main__":
    complex_problem = (
        "Design a microservices migration plan for a legacy monolithic payment service "
        "handling 10,000 transactions/sec, requiring zero downtime and strict ACID compliance."
    )
    run_tot_comparison(complex_problem)