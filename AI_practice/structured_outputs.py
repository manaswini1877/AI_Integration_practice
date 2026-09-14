import os
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
            temperature=0.0

        )
    )
    import json
    return json.loads(response.text)

# --- 2. Test it on something tricky — feedback that mixes multiple signals ---
tricky_feedback = "Honestly the new update is great, love the design! But it crashed twice today which is really annoying."

result = tag_feedback_v2(tricky_feedback)
print("--- Schema-constrained result ---")
print(result)

# --- 3. Stress test: run it 3 times, confirm it NEVER breaks format ---
print("\n--- Reliability check (3 runs, same input) ---")
for i in range(3):
    print(f"Run {i+1}:", tag_feedback_v2(tricky_feedback))