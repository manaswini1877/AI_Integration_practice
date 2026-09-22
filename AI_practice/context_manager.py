import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# We manage history ourselves — a plain list of Content objects
conversation_history = []

def send_message(user_text: str) -> str:
    conversation_history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=conversation_history
    )

    conversation_history.append(types.Content(role="model", parts=[types.Part(text=response.text)]))
    return response.text
import time

def call_with_retry(func, *args, retries=5, delay=15, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                print(f"  (attempt {attempt+1} failed: {e}. Retrying in {delay}s...)")
                time.sleep(delay)
            else:
                raise
# Test: does it actually remember across calls?
print(call_with_retry(send_message, "My name is Manaswini and I'm building an AI feedback tagger."))
print(call_with_retry(send_message, "What's my name, and what am I building?"))