import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
print("\n--- Streaming response ---")
stream = client.models.generate_content_stream(
    model="gemini-3.6-flash",
    contents="Explain what an API is in 3 sentences."
)

for chunk in stream:
    print(chunk.text, end="", flush=True)
print()