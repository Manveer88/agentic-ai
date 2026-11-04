
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Use the same model ID that worked in Lesson 3
MODEL = "gpt-4o-mini"


# -------------------------
# TOOL 1: CALCULATOR
# -------------------------

def calculator(a, b, operation):

    if operation == "add":
        return a + b

    elif operation == "multiply":
        return a * b

    elif operation == "subtract":
        return a - b

    elif operation == "divide":
        if b == 0:
            return "Cannot divide by zero"
        return a / b

    return "Invalid operation"


# -------------------------
# TOOL 2: CURRENT TIME
# -------------------------

def get_current_time():

    return datetime.now().astimezone().isoformat()


# -------------------------
# TOOL 3: WORD COUNT
# -------------------------

def word_count(text):

    return len(text.split())


# -------------------------
# TOOL REGISTRY
# -------------------------

tool_registry = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "word_count": word_count
}


# -------------------------
# TOOL DEFINITIONS
# -------------------------

tools = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Perform mathematical calculations.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "operation": {
                    "type": "string",
                    "enum": [
                        "add", "subtract",
                        "multiply", "divide"
                    ]
                }
            },
            "required": ["a", "b", "operation"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_current_time",
        "description": "Get the current local date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "word_count",
        "description": "Count the number of words in text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"],
            "additionalProperties": False
        },
        "strict": True
    }
]


# -------------------------
# AGENT INSTRUCTIONS
# -------------------------

AGENT_INSTRUCTIONS = """
You are a helpful multi-tool AI agent.

You have three tools:
1. calculator
2. get_current_time
3. word_count

Select the appropriate tool based on the user request.

Use tools when appropriate.
For multi-step tasks, use the result of one
tool in your next decision.

Never invent tool results.
If no tool is needed, answer normally.
"""


# -------------------------
# TOOL EXECUTOR
# -------------------------

def execute_tool(call):

    name = call.name

    if name not in tool_registry:
        return "Error: Unknown tool"

    try:
        arguments = json.loads(call.arguments)

        function = tool_registry[name]

        result = function(**arguments)

        print(f"Tool executed: {name}")
        print(f"Result: {result}")

        return str(result)

    except Exception as error:
        return f"Tool execution error: {error}"


# -------------------------
# AGENT LOOP
# -------------------------

def run_agent(user_task):

    print("\nUser:", user_task)

    response = client.responses.create(
        model=MODEL,
        instructions=AGENT_INSTRUCTIONS,
        input=user_task,
        tools=tools,
        parallel_tool_calls=False
    )

    max_steps = 10

    for step in range(max_steps):

        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            print("\nAgent:", response.output_text)
            return

        outputs = []

        for call in tool_calls:

            print(f"\nAgent step: {step + 1}")
            print("Selected tool:", call.name)

            result = execute_tool(call)

            outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": result
            })

        response = client.responses.create(
            model=MODEL,
            previous_response_id=response.id,
            instructions=AGENT_INSTRUCTIONS,
            input=outputs,
            tools=tools,
            parallel_tool_calls=False
        )

    print("Agent stopped: Maximum steps reached.")


# -------------------------
# INTERACTIVE TERMINAL
# -------------------------

while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Agent: Goodbye!")
        break

    try:
        run_agent(user_input)

    except Exception as error:
        print("Error:", error)
