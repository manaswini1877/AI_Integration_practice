from dotenv import load_dotenv
load_dotenv()
import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
short_text = "Hi!"
long_text = "The app crashed every time I tried to upload a photo. This has been happening for the past three weeks and it's making it impossible for me to complete my work. I've tried reinstalling the app, clearing the cache, and restarting my phone, but nothing has fixed it."

short_response = client.models.count_tokens(model="gemini-3.6-flash", contents=short_text)
long_response = client.models.count_tokens(model="gemini-3.6-flash", contents=long_text)

print("Short text tokens:", short_response.total_tokens)
print("Long text tokens:", long_response.total_tokens)