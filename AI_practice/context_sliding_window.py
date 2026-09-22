import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# We manage history ourselves — a plain list of Content objects
conversation_history = []
MAX_HISTORY_PAIRS = 3  # keep only the last 3 user+model exchanges

def trim_history():
    global conversation_history
    # Each exchange is 2 entries (user + model), so keep the last N*2 entries
    max_entries = MAX_HISTORY_PAIRS * 2
    if len(conversation_history) > max_entries:
        conversation_history = conversation_history[-max_entries:]

def send_message(user_text: str) -> str:
    conversation_history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=conversation_history
    )

    conversation_history.append(types.Content(role="model", parts=[types.Part(text=response.text)]))
    trim_history() 
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
# Send a few more messages, then check the window trimmed old ones
call_with_retry(send_message, "I'm also learning Java and Spring Boot.")
call_with_retry(send_message, "I'm from SVECW, Bhimavaram.")
call_with_retry(send_message, "What programming language did I mention first?")
print("\n--- Answer to 'what language did I mention first?' ---")
final_answer = call_with_retry(send_message, "What programming language did I mention first?")
print(final_answer)

print("\n--- Current history length ---")
print(len(conversation_history), "entries (", len(conversation_history)//2, "exchanges )")