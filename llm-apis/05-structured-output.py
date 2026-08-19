import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

client = genai.Client()

# 1. Define the exact JSON schema using Pydantic
class CharacterProfile(BaseModel):
    name: str
    age: int
    role: str
    skills: list[str] = Field(description="List of key abilities or skills")
    is_hero: bool

prompt = "Generate a main character profile for a sci-fi video game."

# 2. Request structured output using response_schema & response_mime_type
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CharacterProfile,
    ),
)

# 3. Access the raw JSON string or inspect response.text
print("Raw JSON Response:")
print(response.text)