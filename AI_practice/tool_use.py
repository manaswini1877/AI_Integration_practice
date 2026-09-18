import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

create_ticket_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="create_support_ticket",
            description="Creates a support ticket for a customer issue that needs follow-up.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "issue_summary": types.Schema(type=types.Type.STRING, description="Short summary of the issue"),
                    "priority": types.Schema(type=types.Type.STRING, enum=["low", "medium", "high"]),
                },
                required=["issue_summary", "priority"]
            )
        )
    ]
)

user_message = "My payment failed three times and I've been charged each time without getting the product. This is urgent."

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=user_message,
    config=types.GenerateContentConfig(
        tools=[create_ticket_tool]
    )
)

model_part = response.candidates[0].content.parts[0]  # <-- CHANGED: keep the whole part, not just function_call

def create_support_ticket(issue_summary: str, priority: str) -> dict:
    print(f"[TICKET CREATED] Priority: {priority.upper()} | Issue: {issue_summary}")
    return {"status": "created", "ticket_id": "TCK-1029"}

function_call = response.candidates[0].content.parts[0].function_call

if function_call.name == "create_support_ticket":
    result = create_support_ticket(**function_call.args)
    print("Execution result:", result)

final_response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=[
        types.Content(role="user", parts=[types.Part(text=user_message)]),
        types.Content(role="model", parts=[model_part]),  # <-- CHANGED: reuse original part (has thought_signature)
        types.Content(role="user", parts=[types.Part(
            function_response=types.FunctionResponse(
                name="create_support_ticket",
                response=result
            )
        )])
    ],
    config=types.GenerateContentConfig(tools=[create_ticket_tool])
)

print("\n--- Final response to user ---")
print(final_response.text)