

import ollama
import json
import os

# Always use the same file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "conversation_history.json")


MEMORY_FILE = os.path.join(BASE_DIR, "important_memory.json")


def load_important_memory():

    if not os.path.exists(MEMORY_FILE):
        return {}

    try:
        with open(MEMORY_FILE, "r") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return {}


def save_important_memory(memory):

    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)

# Load conversation history
def load_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except (json.JSONDecodeError, OSError):
        return []


# Save conversation history
def save_history():

    with open(HISTORY_FILE, "w") as file:
        json.dump(
            conversation_history,
            file,
            indent=4
        )


# Initialize memory ONCE
conversation_history = load_history()

print(
    f"[Memory] Loaded {len(conversation_history)} messages"
)



def get_recent_history(limit=10):

    return conversation_history[-limit:]


# Tool 1: Search company knowledge base

def search_knowledge_base(question=None, **kwargs):

    print("\n[Agent] Searching company knowledge base...")

    if not question:
        return "No search question provided."

    from lesson_07 import retrieve_documents

    documents = retrieve_documents(question)

    return json.dumps(documents, indent=2)



# Tool 2: Get current time

def get_current_time(**kwargs):

    from datetime import datetime

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# Tool registry
tools = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time
}


def detect_intents(message):

    message = message.lower()

    intents = []

    knowledge_keywords = [
    "refund",
    "policy",
    "support",
    "account",
    "password",
    "reset",
    "receipt",
    "order",
    "email",
    "document",
    "documents",
    "working hours",
    "payment"
]

    time_keywords = [
        "time", "date", "today",
        "current time"
    ]

    if any(word in message for word in knowledge_keywords):
        intents.append("knowledge")

    if any(word in message for word in time_keywords):
        intents.append("time")

    if not intents:
        intents.append("general")

    return intents


def is_greeting(message):

    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you"
    ]

    normalized = message.lower().strip()
    normalized = normalized.rstrip("!.,?")

    return normalized in greetings

# Tools that the LLM can see and choose

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search company documents for information about company policies, refunds, support, accounts and passwords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to search in company documents"
                    }
                },
                "required": ["question"]
            }
        }
    },    
    {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Return the current date and time. This function takes no arguments.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}

]




def run_agent(user_message):

    global conversation_history

    # 1. Detect intent
    intents = detect_intents(user_message)

    print(f"\n[Router] Detected intents: {intents}")

    # 2. Prepare search query with conversation context
    search_query = user_message

    if "knowledge" in intents:

        print("\n[Agent] Executing knowledge search")

        # Include recent conversation when resolving follow-ups
        previous_context = "\n".join(
            item["content"]
            for item in conversation_history[-6:]
        )

        search_query = (
            previous_context + "\n" + user_message
        )

        result = search_knowledge_base(search_query)

        tool_context = f"COMPANY INFORMATION:\n{result}"

    else:
        tool_context = ""

    # 3. Execute time tool
    if "time" in intents:

        print("\n[Agent] Executing time tool")

        current_time = get_current_time()

        tool_context += (
            f"\nCURRENT DATE AND TIME:\n{current_time}"
        )

    # 4. Prepare current message
    current_message = user_message

    if tool_context:
        current_message += (
            "\n\nRelevant information:\n"
            + tool_context
        )

    # Keep the full context for the current LLM request,
    # but store only the original user question in history.
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    save_history()

    # 5. Send conversation history to LLM
    messages = [
        {
            "role": "system",
            "content": """
            You are a company knowledge assistant.

            Remember previous conversation messages.

            Use retrieved company information to answer
            company-policy questions.

            Never invent company policies.
            """
        }
    ]

    messages.extend(get_recent_history(10))

    messages.append({
        "role": "user",
        "content": current_message
    })

    response = ollama.chat(
        model="llama3.2:3b",
        messages=messages
    )

    # 6. Save AI response
    ai_answer = response.message.content

    conversation_history.append({
        "role": "assistant",
        "content": ai_answer
    })

    save_history()

    print("\nAI:", ai_answer)




while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    if question.lower() == "reset":
        conversation_history.clear()
        save_history()

        print("\n[Memory] Conversation history cleared.")

        continue

    if question.strip():
        try:
            run_agent(question)

        except Exception as error:
            print("Error:", error)


print("Agent tools registered:")
for tool_name in tools:
    print("-", tool_name)


