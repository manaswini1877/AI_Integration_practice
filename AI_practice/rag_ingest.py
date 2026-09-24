import os
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

with open("notes.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(chunks)} chunks")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_notes")

for i, chunk in enumerate(chunks):
    embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    ).embeddings[0].values

    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk]
    )

print("All chunks embedded and stored in Chroma.")