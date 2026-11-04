
import ollama
import json






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
        "refund", "policy", "support",
        "account", "password", "reset",
        "receipt", "order", "email"
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

# def run_agent(user_message):

#     messages = [
#         {
#             "role": "system",
#             "content": """
#             You are a helpful AI agent.

#             You have access to two tools:
#             1. Search company knowledge base.
#             2. Get current time.

#             Use the knowledge search tool for questions
#             about company policies and information.

#             Use the time tool when the user asks about
#             current date or time.

#             For greetings and general conversation,
#             answer directly without using tools.

#             Never invent company policies.
#             """
#         },
#         {
#             "role": "user",
#             "content": user_message
#         }
#     ]

#     response = ollama.chat(
#         model="llama3.2:3b",
#         messages=messages,
#         tools=tool_definitions
#     )

#     # Check whether the AI requested a tool
#     for tool_call in response.message.tool_calls or []:

#         function_name = tool_call.function.name
#         arguments = tool_call.function.arguments

#         print(f"\n[Agent] Calling tool: {function_name}")

#         function_to_call = tools.get(function_name)

#         if function_to_call:

#             if function_name == "search_knowledge_base" and not isinstance(arguments.get("question"), str):
#                 arguments["question"] = user_message

#             result = function_to_call(**arguments)

#             messages.append(response.message)

#             messages.append({
#                 "role": "tool",
#                 "name": function_name,
#                 "content": str(result)
#             })

#             # Ask AI to generate final response
#             response = ollama.chat(
#                 model="llama3.2:3b",
#                 messages=messages,
#                 tools=tool_definitions
#             )

#     print("\nAI:", response.message.content)



def run_agent(user_message):

    # 1. Detect intents
    intents = detect_intents(user_message)

    print(f"\n[Router] Detected intents: {intents}")

    # 2. Execute tools deterministically
    tool_results = []

    if "knowledge" in intents:

        print("\n[Agent] Executing knowledge search")

        result = search_knowledge_base(user_message)

        tool_results.append(
            f"COMPANY INFORMATION:\n{result}"
        )

    if "time" in intents:

        print("\n[Agent] Executing time tool")

        result = get_current_time()
        print("[Time Tool Result]:", result)
        tool_results.append(
            f"CURRENT DATE AND TIME:\n{result}"
        )

    # 3. Prepare context
    context = "\n\n".join(tool_results)

    # 4. Ask LLM to generate final answer
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
    "role": "system",
    "content": """
    You are a helpful company AI assistant.

    IMPORTANT RULES:

    1. Answer company questions using COMPANY INFORMATION.
    2. If CURRENT DATE AND TIME is provided,
       explicitly include it when the user asks for the time.
    3. Never confuse support working hours with current time.
    4. Never invent company policies.
    5. If information is missing, clearly say so.
    """
},
           {
        "role": "user",
        "content": f"""
        User question:
        {user_message}

        Tool results:
        {context}

        Answer every part of the user's question.
        Include the current date and time if requested.
        """
    }
        ]
    )

    print("\nAI:", response.message.content)



while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    if question.strip():
        run_agent(question)


print("Agent tools registered:")
for tool_name in tools:
    print("-", tool_name)


