from dotenv import load_dotenv
load_dotenv()
import os
import math
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
sentence1 = "The app crashed during photo upload."
sentence2 = "I can't upload pictures, the app keeps freezing."
sentence3 = "I love the new dark mode design."

emb1 = client.models.embed_content(model="gemini-embedding-001", contents=sentence1).embeddings[0].values
emb2 = client.models.embed_content(model="gemini-embedding-001", contents=sentence2).embeddings[0].values
emb3 = client.models.embed_content(model="gemini-embedding-001", contents=sentence3).embeddings[0].values
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    return dot_product / (magnitude_a * magnitude_b)

print("Sentence 1 vs 2 (similar meaning):", cosine_similarity(emb1, emb2))
print("Sentence 1 vs 3 (different topic):", cosine_similarity(emb1, emb3))