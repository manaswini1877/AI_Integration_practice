
from dotenv import load_dotenv
load_dotenv()
import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
response = client.models.count_tokens(
    model="gemini-3.6-flash",
    contents="The app crashed every time I tried to upload a photo."
)
print(response)