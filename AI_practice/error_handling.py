import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def robust_generate(prompt: str, primary_model="gemini-3.6-flash", fallback_model="gemini-2.5-flash", max_retries=4) -> str:
    models_to_try = [primary_model, fallback_model]

    for model_name in models_to_try:
        delay = 2  # start at 2 seconds, doubles each retry (exponential backoff)
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        http_options=types.HttpOptions(timeout=10000)  # 10 second timeout, in ms
                    )
                )
                if model_name != primary_model:
                    print(f"  (note: had to fall back to {model_name})")
                return response.text

            except Exception as e:
                error_str = str(e)
                is_last_attempt = attempt == max_retries - 1

                if is_last_attempt:
                    print(f"  {model_name} exhausted all {max_retries} attempts, trying next model...")
                    break  # move to fallback model

                print(f"  ({model_name} attempt {attempt+1} failed: {error_str[:80]}... retrying in {delay}s)")
                time.sleep(delay)
                delay *= 2  # exponential backoff: 2s -> 4s -> 8s -> 16s

    raise Exception("All models and retries exhausted — could not get a response.")


# Test it
result = robust_generate("Give me one sentence explaining what a sliding window is in AI context management.")
print("\n--- Final result ---")
print(result)