import os
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

# Define nested Pydantic schemas
class MicroserviceSpec(BaseModel):
    name: str = Field(description="Name of the service, e.g. auth-service")
    language: str = Field(description="Programming language used")
    port: int = Field(description="Port number the service listens on")
    dependencies: List[str] = Field(description="List of database or cache dependencies")

class ArchitectureBlueprint(BaseModel):
    project_name: str
    architecture_style: str = Field(description="e.g. Microservices, Serverless, Monolith")
    estimated_monthly_cost_usd: float
    services: List[MicroserviceSpec]

def generate_architecture_schema(user_request: str) -> ArchitectureBlueprint:
    """Generates structured architectural data adhering strictly to the Pydantic schema."""
    client = genai.Client()

    prompt = f"Design a backend infrastructure for the following app request:\n{user_request}"

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ArchitectureBlueprint,
            temperature=0.2,  # Low temperature ensures deterministic adherence
        )
    )

    # Validate and convert raw JSON string directly to Pydantic instance
    blueprint = ArchitectureBlueprint.model_validate_json(response.text)
    return blueprint

if __name__ == "__main__":
    app_request = "An e-commerce system with an authentication service, payment gateway, and product catalog."

    print("Sending request for Structured Output...\n")
    result = generate_architecture_schema(app_request)

    print(f"Project: {result.project_name}")
    print(f"Style: {result.architecture_style}")
    print(f"Estimated Cost: ${result.estimated_monthly_cost_usd}/mo")
    print("\nServices Defined:")
    for svc in result.services:
        print(f"  - {svc.name} ({svc.language}) on port {svc.port} | Deps: {', '.join(svc.dependencies)}")