import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
prompt = "Give me a one-sentence tagline for a fitness app."

low_temp = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
    config=types.GenerateContentConfig(temperature=0.0)
)

high_temp = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
    config=types.GenerateContentConfig(temperature=1.5)
)

print("\n--- Low temperature (0.0) ---")
print(low_temp.text)
print("\n--- High temperature (1.5) ---")
print(high_temp.text)