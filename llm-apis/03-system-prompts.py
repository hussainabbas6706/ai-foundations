import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API credentials from the .env file
load_dotenv()

# Define three completely different system personas
SYSTEM_PROMPTS = {
    "Pirate": "You are a salty 17th-century pirate captain. Explain everything using pirate slang, sea shanty references, and nautical terms.",
    "ELI5 (Explain Like I'm 5)": "You are a patient kindergarten teacher. Explain using simple words, fun analogies, and a warm tone suitable for a 5-year-old.",
    "Dry Technical Spec": "You are an extremely precise and dry computer science reference manual. Avoid conversational filler. List only technical definitions and data architecture terms."
}

def run_experiment():
    print("Initializing Gemini Client...")
    try:
        # Initialize the client (automatically reads GEMINI_API_KEY from environment)
        client = genai.Client()
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return

    user_query = "What is a database?"
    print(f"User Query: \"{user_query}\"\n")
    print("=" * 60)

    # Loop through each persona, generate content, and print results
    for persona, sys_instruction in SYSTEM_PROMPTS.items():
        print(f"\n[System Persona: {persona}]")
        print(f"Instruction: \"{sys_instruction}\"\n")
        
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=user_query,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    temperature=0.7,
                )
            )
            print(response.text.strip())
        except Exception as e:
            print(f"Error running model for {persona}: {e}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    run_experiment()
