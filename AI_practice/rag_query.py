import os
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Connect to the SAME Chroma database you just built
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_notes")

def ask(question: str) -> str:
    # 1. RETRIEVE: embed the question, find the most similar stored chunks
    question_embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    ).embeddings[0].values

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=2  # get the top 2 most relevant chunks
    )
    retrieved_chunks = results["documents"][0]

    print("--- Retrieved chunks ---")
    for chunk in retrieved_chunks:
        print("-", chunk)

    # 2. AUGMENT: build a prompt that includes the retrieved facts as context
    context = "\n".join(retrieved_chunks)
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}
"""

    # 3. GENERATE: ask the model, grounded in the retrieved context
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text

# Test it
print("\n--- Answer ---")
print(ask("What does RAG stand for?"))
print(ask("What is my name?"))