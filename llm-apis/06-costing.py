import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3.5-flash")

prompt = "Explain RAG in 3 lines."

# Count tokens BEFORE sending
token_count = model.count_tokens(prompt)
print(f"Estimated tokens before call: {token_count.total_tokens}")

# Make the actual call
response = model.generate_content(prompt)

# Token usage after call
print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
print(f"Completion tokens: {response.usage_metadata.candidates_token_count}")
print(f"Total tokens: {response.usage_metadata.total_token_count}")
            
# Gemini 3.5 Flash pricing
# Input: $0.10 per 1M tokens
# Output: $0.40 per 1M tokens
prompt_cost = (response.usage_metadata.prompt_token_count / 1_000_000) * 0.10
completion_cost = (response.usage_metadata.candidates_token_count / 1_000_000) * 0.40
total_cost = prompt_cost + completion_cost

print(f"\nEstimated cost: ${total_cost:.6f}")
print(f"Gemini is {'cheaper' if total_cost < 0.001 else 'reasonable'} for this call")