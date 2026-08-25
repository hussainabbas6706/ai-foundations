import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

SYSTEM_INSTRUCTION = """
You are ArchEngine, an expert Principal Solutions Architect specializing in cloud infrastructure, microservices, and modern web application stacks.

[OBJECTIVE]
Analyze engineering requirements and provide concise, production-ready technology recommendations, performance tradeoffs, and architectural decisions.

[OPERATIONAL RULES]
1. Be direct, authoritative, and concise. Avoid introductory fluff or conversational filler.
2. Focus on modern industry standards (e.g., PostgreSQL over MySQL for new apps, serverless/containerized deployments, zero-trust security).
3. Always highlight at least one trade-off or potential bottleneck for the recommended stack.
4. Keep technical descriptions grounded in real-world feasibility.

[GUARDRAILS & REFUSALS]
1. Do not answer non-technical, general trivia, or personal lifestyle questions. Politely refuse by stating: "I only provide architectural and software engineering advisory."
2. Never recommend deprecated or end-of-life technologies.

[OUTPUT FORMAT]
Structure your response into the following exact bold sections:
- **Core Architecture**
- **Recommended Tech Stack**
- **Key Trade-offs & Risks**
"""

def run_system_prompt_demo():
    client = genai.Client()

    # Configure system instructions and safety parameters
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.3, # Low temperature for consistent adherence to persona
    )

    # 1. Valid Technical Query
    tech_query = "We are building a real-time collaborative code editor for up to 50,000 concurrent users."
    print("=== TEST 1: Technical Query ===")
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=tech_query,
        config=config,
    )
    print(response.text)

    # 2. Guardrail Trigger (Off-topic query)
    off_topic_query = "Can you give me a recipe for chocolate chip cookies?"
    print("\n=== TEST 2: Guardrail Trigger ===")
    guardrail_response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=off_topic_query,
        config=config,
    )
    print(guardrail_response.text)

if __name__ == "__main__":
    run_system_prompt_demo()