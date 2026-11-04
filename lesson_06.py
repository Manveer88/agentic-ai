
import json
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

MODEL = "gpt-4o-mini"

MEMORY_FILE = "memory.json"

# -------------------------
# LONG-TERM MEMORY
# -------------------------

def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return {}

    with open(MEMORY_FILE, "r") as file:
        return json.load(file)


def save_memory(memory):

    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)


memory = load_memory()


# -------------------------
# AGENT CONFIGURATION
# -------------------------

previous_response_id = None

def run_agent(user_input):

    global previous_response_id, memory

    if user_input.startswith("/remember "):

        fact = user_input.replace("/remember ", "", 1)

        if "=" not in fact:
            print("Use: /remember key=value")
            return

        key, value = fact.split("=", 1)

        memory[key.strip()] = value.strip()

        save_memory(memory)

        print("Agent: Memory saved!")
        return

    if user_input == "/memory":

        print("Saved memory:", json.dumps(
            memory, indent=4
        ))
        return

    if user_input.startswith("/forget "):

        key = user_input.replace("/forget ", "", 1).strip()

        if key in memory:
            del memory[key]
            save_memory(memory)
            print("Agent: Memory forgotten.")
        else:
            print("Agent: No such memory found.")

        return

    # Pass long-term memory to the LLM
    instructions = f"""
    You are a helpful AI assistant.

    Use the following saved user information
    when it is relevant.

    SAVED MEMORY:
    {json.dumps(memory, indent=2)}

    Do not invent personal information.
    If information is not available, say so.
    """

    request = {
        "model": MODEL,
        "instructions": instructions,
        "input": user_input
    }

    # SHORT-TERM MEMORY
    if previous_response_id:
        request["previous_response_id"] = (
            previous_response_id
        )

    response = client.responses.create(**request)

    previous_response_id = response.id

    print("\nAgent:", response.output_text)


# -------------------------
# INTERACTIVE CHAT
# -------------------------

print("Memory Agent Started!")
print("Commands: /remember, /memory, /forget, exit")

while True:

    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Agent: Goodbye!")
        break

    if not user_input:
        continue

    try:
        run_agent(user_input)

    except Exception as error:
        print("Error:", error)
