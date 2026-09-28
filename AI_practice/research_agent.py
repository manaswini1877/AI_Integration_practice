import os
import json
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Tool 1: a fake "search" tool (simulated — returns canned facts, no real internet call yet)
def search_web(query: str) -> str:
    fake_database = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "cloud computing": "Cloud computing delivers computing services over the internet, including servers, storage, and databases.",
        "rag": "RAG (Retrieval-Augmented Generation) combines document retrieval with LLM generation for grounded answers.",
    }
    for key, value in fake_database.items():
        if key in query.lower():
            return value
    return "No results found for that query."

def calculate(expression: str) -> str:
    # Only allow digits, basic operators, parentheses, spaces, decimal points
    if not re.fullmatch(r"[0-9+\-*/(). %]+", expression):
        return "Error: only basic math expressions are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

# Describe both tools to the model
tools = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="search_web",
        description="Search for information on a topic.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"query": types.Schema(type=types.Type.STRING)},
            required=["query"]
        )
    ),
    types.FunctionDeclaration(
        name="calculate",
        description="Evaluate a math expression, e.g. '15 * 3'.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"expression": types.Schema(type=types.Type.STRING)},
            required=["expression"]
        )
    ),
])

# Map tool names to the real Python functions
available_functions = {
    "search_web": search_web,
    "calculate": calculate,
}

def run_agent(question: str, max_steps: int = 5) -> str:
    history = [types.Content(role="user", parts=[types.Part(text=question)])]

    for step in range(max_steps):
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=history,
            config=types.GenerateContentConfig(tools=[tools])
        )

        # Save the model's full turn (this keeps the thought_signature, the bug from Topic 6)
        model_content = response.candidates[0].content
        history.append(model_content)

        # Did the model ask for any tools this turn?
        function_calls = [p.function_call for p in model_content.parts if p.function_call]

        # No tool requested = the model is done, this is the final answer
        if not function_calls:
            return response.text

        # Otherwise: run every requested tool and send the results back
        result_parts = []
        for fc in function_calls:
            print(f"[Step {step+1}] Model called {fc.name}({dict(fc.args)})")
            result = available_functions[fc.name](**fc.args)
            print(f"           Result: {result}")
            result_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": result}
                )
            ))
        history.append(types.Content(role="user", parts=result_parts))

    return "Stopped: reached max steps without a final answer."

# Test: this question needs BOTH tools
answer = run_agent("Search for who created Python and in what year, then calculate how many years ago that was from 2026.")
print("\n--- Final answer ---")
print(answer)