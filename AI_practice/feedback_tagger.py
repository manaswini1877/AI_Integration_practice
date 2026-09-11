import os
import json
from google import genai

# Set up the client using your API key stored as an environment variable
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def tag_feedback(text: str) -> dict:
    # The prompt: tells the model exactly what to analyze and what format to reply in
    prompt = f"""Analyze this customer feedback and respond with ONLY valid JSON, no extra text.

Feedback: "{text}"

Return JSON in exactly this format:
{{"sentiment": "positive/negative/neutral", "urgency": "low/medium/high", "category": "bug/feature request/praise/other"}}
"""

    # Send the prompt to the model and get a response back
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    # response.text is a STRING (even though it looks like JSON) —
    # json.loads() converts that string into a real Python dict
    result = json.loads(response.text)
    return result


# --- test input ---
sample_feedback = "The app crashed every time I tried to upload a photo. Really frustrating!"

result = tag_feedback(sample_feedback)
print(result)
print(tag_feedback("I love the new dashboard, it's so clean!"))
print(tag_feedback("Can you add dark mode? Would be nice."))