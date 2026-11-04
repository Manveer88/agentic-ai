import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def ask_ai(question):

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="""
        You are a helpful AI programming tutor.
        Explain concepts with simple examples.
        """,
        input=question
    )

    return response.output_text


if __name__ == "__main__":

    question = input("Ask something: ")

    answer = ask_ai(question)

    print("\nAI Response:")
    print(answer)