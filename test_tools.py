
import ollama

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "What time is it? Use the get_current_time tool."
        }
    ],
    tools=tools
)

print("Response content:")
print(response.message.content)

print("\nTool calls:")
print(response.message.tool_calls)
