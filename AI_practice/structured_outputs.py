import os
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# --- 1. Define the exact shape we require, using a schema ---
feedback_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "sentiment": types.Schema(type=types.Type.STRING, enum=["positive", "negative", "neutral"]),
        "urgency": types.Schema(type=types.Type.STRING, enum=["low", "medium", "high"]),
        "category": types.Schema(type=types.Type.STRING, enum=["bug", "feature request", "praise", "other"]),
    },
    required=["sentiment", "urgency", "category"]
)

def tag_feedback_v2(text: str) -> dict:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"Analyze this customer feedback: {text}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=feedback_schema,
            system_instruction="""Classify feedback using these exact rules:
- urgency=high ONLY if the app is completely broken/unusable or data was lost
- urgency=medium if there's a real bug but the app is still partially usable
- urgency=low if it's a minor issue, suggestion, or general feedback
- If feedback mixes praise and a complaint, sentiment reflects the complaint if it describes a functional problem, otherwise neutral
Apply these rules consistently — do not deviate based on tone or wording."""
        )
    )
    return json.loads(response.text)

# --- NEW: retry wrapper for transient failures like 503s ---
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

# --- 2. Test it on something tricky — feedback that mixes multiple signals ---
tricky_feedback = "Honestly the new update is great, love the design! But it crashed twice today which is really annoying."

result = call_with_retry(tag_feedback_v2, tricky_feedback)
print("--- Schema-constrained result ---")
print(result)

# --- 3. Stress test: run it 3 times, confirm it NEVER breaks format ---
print("\n--- Reliability check (3 runs, same input) ---")
for i in range(3):
    print(f"Run {i+1}:", call_with_retry(tag_feedback_v2, tricky_feedback))