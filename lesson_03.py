
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# 1. Define our Python tool
def calculator(a, b, operation):

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":
        if b == 0:
            return "Cannot divide by zero"
        return a / b

    return "Invalid operation"


# 2. Tell the LLM which tool is available
tools = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Perform mathematical calculations.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "add",
                        "subtract",
                        "multiply",
                        "divide"
                    ]
                }
            },
            "required": ["a", "b", "operation"],
            "additionalProperties": False
        },
        "strict": True
    }
]


# 3. Send user request to LLM
user_input = input("You: ")

response = client.responses.create(
    model="gpt-4o-mini",
    input=user_input,
    tools=tools
)


# 4. Check whether the LLM requested a tool
input_list = [
    {"role": "user", "content": user_input}
]

input_list += response.output

for item in response.output:

    if item.type == "function_call":

        if item.name == "calculator":

            arguments = json.loads(item.arguments)

            # Execute Python function
            result = calculator(**arguments)

            print("Tool executed!")
            print("Tool result:", result)

            # Send result back to LLM
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": str(result)
            })


# 5. Get final response
if any(item.type == "function_call" for item in response.output):

    final_response = client.responses.create(
      model="gpt-4o-mini",
        tools=tools,
        input=input_list
    )

    print("AI Agent:", final_response.output_text)

else:
    print("AI Agent:", response.output_text)
