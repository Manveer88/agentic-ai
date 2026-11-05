
import ollama
import json
from lesson_09 import (
    search_knowledge_base,
    get_current_time,
    load_important_memory
)

MODEL = "llama3.2:3b"

ALLOWED_TOOLS = {
    "search_knowledge_base",
    "get_current_time"
    }

MAX_STEPS = 5


def validate_plan(plan):

    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")

    steps = plan.get("steps")

    if not isinstance(steps, list):
        raise ValueError("Plan must contain a steps list.")

    if len(steps) == 0:
        raise ValueError("Plan cannot be empty.")

    if len(steps) > MAX_STEPS:
        raise ValueError("Too many execution steps.")

    for step in steps:

        if not isinstance(step, dict):
            raise ValueError("Each step must be an object.")

        tool_name = step.get("tool")
        description = step.get("description")

        if tool_name not in ALLOWED_TOOLS:
            raise ValueError(f"Unauthorized tool: {tool_name}")

        if not isinstance(description, str) or not description.strip():
            raise ValueError("Invalid step description.")

    return True

def create_plan(user_task):

    prompt = f"""
    You are an AI task planner.

    Available tools (ONLY these two):
    - search_knowledge_base
    - get_current_time

    Rules:
    1. You MUST use only the exact tool names listed above.
    2. NEVER create new tools.
    3. For company information, policies, refunds, support,
    account details and passwords, use search_knowledge_base.
    4. For current time or date, use get_current_time.
    5. For greetings or general conversation, do not create a plan.
    6. Maximum 5 steps.
    7. Return only valid JSON.

    Required format:
    {{
        "steps": [
            {{
                "tool": "search_knowledge_base",
                "description": "Search refund policy"
            }}
        ]
    }}

    User task: {user_task}
    """

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json"
    )

    try:
        plan = json.loads(response.message.content)
        return plan

    except json.JSONDecodeError:
        raise ValueError(
            "Planner returned invalid JSON. Please try again."
        )


def execute_plan(plan):

    results = []

    print("\n[Executor] Starting execution...")

    for index, step in enumerate(plan["steps"], start=1):

        tool_name = step["tool"]
        description = step["description"]

        print(f"\n[Executor] Step {index}: {description}")

        try:
            if tool_name == "search_knowledge_base":
                result = search_knowledge_base(description)

            elif tool_name == "get_current_time":
                result = get_current_time()

            else:
                raise ValueError(f"Unsupported tool: {tool_name}")

            results.append({
                "step": index,
                "tool": tool_name,
                "status": "success",
                "result": result
            })

            print(f"[Executor] Step {index} completed.")

        except Exception as error:

            print(f"[Executor] Step {index} failed: {error}")

            results.append({
                "step": index,
                "tool": tool_name,
                "status": "failed",
                "error": str(error)
            })

    return results


def generate_final_answer(user_task, results):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful AI assistant.

                Summarize the executed task results.
                Do not invent information.
                Clearly mention if a step failed.
                """
            },
            {
                "role": "user",
                "content": f"""
               User task: {user_task}
                Execution results:
                {json.dumps(results, indent=4)}

                Instructions:
                - Summarize successful tool results.
                - Clearly mention failed steps.
                - Never invent missing information.
                - If a tool failed, explain that its information could not be retrieved.
                """
            }
        ]
    )

    return response.message.content


def run_planning_agent(user_task):

    print("\n[Planner] Creating execution plan...")

    try:
        plan = create_plan(user_task)

        try:
            validate_plan(plan)
        except ValueError as error:
            print("[Planner] Invalid plan:", error)
            print("[Planner] Trying again...")

        plan = create_plan(user_task)
        validate_plan(plan)

        print("[Validator] Plan approved.")


        print("\n[Planner] Generated plan:")

        print(json.dumps(plan, indent=2))

        results = execute_plan(plan)

        print("\n[Agent] Generating final answer...")

        answer = generate_final_answer(
            user_task,
            results
        )

        print("\nFinal Answer:")
        print(answer)

    except Exception as error:
        print("\nAgent Error:", error)


while True:

    task = input("\nYour task: ")

    if task.lower() in ["exit", "quit"]:
        break

    if task.strip():
        run_planning_agent(task)

test_plan = {
    "steps": [
        {
            "tool": "delete_database",
            "description": "Delete database"
        }
    ]
}

try:
    validate_plan(test_plan)
except ValueError as error:
    print("Validation blocked:", error)