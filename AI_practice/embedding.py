from dotenv import load_dotenv
load_dotenv()
import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
embedding_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="The app crashed during photo upload."
)

vector = embedding_response.embeddings[0].values
print("Vector length:", len(vector))
print("First 5 numbers:", vector[1:5])