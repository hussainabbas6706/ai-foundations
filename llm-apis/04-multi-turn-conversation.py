import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Initialize the client (automatically reads GEMINI_API_KEY from environment)
client = genai.Client()

# Create a chat session with system instructions and a model
chat = client.chats.create(
    model="gemini-3.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful AI assistant."
    )
)

print("Chat started. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        break
    
    # Send message to the ongoing chat session
    response = chat.send_message(user_input)
    
    print(f"\nAssistant: {response.text}")
    print(f"[Total tokens so far: {response.usage_metadata.total_token_count}]\n")