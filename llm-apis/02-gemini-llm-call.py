import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from a .env file if present
load_dotenv()

def main():
    print("Sending request to Google Gemini API...")
    
    try:
        # Initialize the Gemini client.
        # It automatically retrieves the GEMINI_API_KEY environment variable.
        client = genai.Client()
        
        # Make a model generation call.
        # We use 'gemini-3.5-flash' as it is fast, free to use in AI Studio, and highly capable.
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents="Explain the concept of 'prompt engineering' in one sentence.",
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful and concise assistant.",
                temperature=0.7,
            )
        )
        
        # Extract and print the response content
        print("\n--- Response from Gemini ---")
        print(response.text)
        print("----------------------------")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Make sure you have set the GEMINI_API_KEY environment variable in your environment or a .env file.")

if __name__ == "__main__":
    main()
