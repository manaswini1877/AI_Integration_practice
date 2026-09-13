import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# --- 1. ZERO-SHOT: no examples, just ask ---
zero_shot_prompt = "Classify the sentiment of this review: 'The battery dies way too fast.'"

zero_shot = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=zero_shot_prompt
)
print("--- Zero-shot ---")
print(zero_shot.text)

# --- 2. FEW-SHOT: show examples of the exact format we want first ---
few_shot_prompt = """Classify sentiment as exactly one word: Positive, Negative, or Neutral.

Review: "Great battery life, lasts all day."
Sentiment: Positive

Review: "Screen cracked after one drop, terrible build."
Sentiment: Negative

Review: "The battery dies way too fast."
Sentiment:"""

few_shot = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=few_shot_prompt
)
print("\n--- Few-shot ---")
print(few_shot.text)

# --- 3. CHAIN-OF-THOUGHT: force step-by-step reasoning ---
cot_prompt = """A store has 120 apples. They sell 45% on Monday and 30 apples on Tuesday.
How many apples are left? Think step by step, showing each calculation, then give the final answer."""

cot = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=cot_prompt
)
print("\n--- Chain-of-thought ---")
print(cot.text)

# --- 4. STRUCTURED PROMPTING: labeled sections so the model can't confuse instructions vs data ---
structured_prompt = """### Task
Extract the product name and issue type from the customer message below.

### Output Format
Product: <name>
Issue: <bug/billing/feature request/other>

### Customer Message
"My CloudSync Pro subscription charged me twice this month, please fix this."
"""

structured = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=structured_prompt
)
print("\n--- Structured prompting ---")
print(structured.text)

# --- 5. SYSTEM PROMPT PATTERN: stable rules in system_instruction, variable input in contents ---
SYSTEM_RULES = """You are a product support classifier.
Always respond in exactly this format, nothing else:
Category: <one word>
Confidence: <High/Medium/Low>"""

def classify_message(user_message: str) -> str:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user_message,          # <- only the variable part changes per call
        config=types.GenerateContentConfig(system_instruction=SYSTEM_RULES)
    )
    return response.text

print("\n--- System prompt pattern (reusable function) ---")
print(classify_message("The app won't let me log in anymore."))
print(classify_message("I just want to say the new update is amazing!"))