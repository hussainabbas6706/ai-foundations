import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file if present
load_dotenv()

def main():
    print("Sending request to OpenAI API...")
    
    try:
        # Initialize the OpenAI client.
        # It automatically retrieves the OPENAI_API_KEY environment variable.
        client = OpenAI()
        
        # Create a chat completion request.
        # We use 'gpt-4o-mini' as it is fast, cost-effective, and highly capable for most tasks.
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful and concise assistant."},
                {"role": "user", "content": "Explain the concept of 'prompt engineering' in one sentence."}
            ],
            temperature=0.7,
        )
        
        # Extract and print the response content
        reply = response.choices[0].message.content
        print("\n--- Response from LLM ---")
        print(reply)
        print("-------------------------")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Make sure you have set the OPENAI_API_KEY environment variable in your environment or a .env file.")

if __name__ == "__main__":
    main()
