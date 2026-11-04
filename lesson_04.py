
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

MODEL = "gpt-4o-mini"

AGENT_INSTRUCTIONS = (
            "You are a calculator agent. "
            "Use the calculator tool for all arithmetic. "
            "For multi-step calculations, complete one step, "
            "observe its result, and then perform the next step."
        )

# 1. Calculator tool
def calculator(a, b, operation):

    if operation == "add":
        return a + b

    elif operation == "multiply":
        return a * b

    else:
        raise ValueError("Unsupported operation")


# 2. Describe tool for the LLM
tools = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Add or multiply two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "multiply"]
                }
            },
            "required": ["a", "b", "operation"],
            "additionalProperties": False
        },
        "strict": True
    }
]


# 3. Start the agent
def run_agent(user_task):

    print("\nUser:", user_task)

    response = client.responses.create(
        model=MODEL,
        instructions=AGENT_INSTRUCTIONS,
        input=user_task,
        tools=tools,
        parallel_tool_calls=False
    )

    # 4. Agent loop
    max_steps = 5

    for step in range(max_steps):

        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        # No more tool calls: agent is finished
        if not tool_calls:
            print("\nAgent:", response.output_text)
            return

        outputs = []

        for call in tool_calls:

            print(f"\nStep {step + 1}: Calling {call.name}")

            if call.name != "calculator":
                raise ValueError("Unknown tool")

            arguments = json.loads(call.arguments)

            # Basic argument validation
            if (
                set(arguments) != {"a", "b", "operation"}
                or type(arguments["a"]) not in (int, float)
                or type(arguments["b"]) not in (int, float)
                or arguments["operation"] not in ("add", "multiply")
            ):
                raise ValueError("Invalid tool arguments")

            # Execute actual Python function
            result = calculator(**arguments)

            print("Tool result:", result)

            outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": str(result)
            })

        # Send tool results back to the LLM
        response = client.responses.create(
            model=MODEL,
            previous_response_id=response.id,
            input=outputs,
            tools=tools,
            parallel_tool_calls=False
        )

    print("Agent stopped: Maximum steps reached.")


# 5. Interactive terminal
while True:

    task = input("\nYou: ")

    if task.lower() in ["exit", "quit"]:
        print("Agent: Goodbye!")
        break

    try:
        run_agent(task)
    except Exception as error:
        print("Error:", error)
