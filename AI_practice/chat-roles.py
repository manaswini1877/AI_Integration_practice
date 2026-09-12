import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="My app keeps crashing, what should I do?",
    config=types.GenerateContentConfig(
        system_instruction="You are a terse, no-nonsense tech support agent. Max 2 sentences"
    )
)
print(response.text)
response2 = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="My app keeps crashing, what should I do?",
    config=types.GenerateContentConfig(
        system_instruction="You are a warm, empathetic support agent. Reassure the user and explain steps gently."
    )
)
print("\n--- Warm persona ---")
print(response2.text)