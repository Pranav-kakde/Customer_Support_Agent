import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def get_llm_response(
    system_prompt,
    conversation,
    context=""
):

    contents = []


    for message in conversation:

        role = message["role"]
        content = message["content"]


        if role == "user":

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=content
                        )
                    ]
                )
            )


        elif role == "assistant":

            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=content
                        )
                    ]
                )
            )


    # Add retrieved context to the latest user question
    if context:

        last_user_message = conversation[-1]["content"]

        rag_prompt = f"""
Use the following knowledge base context to answer
the customer's question.

KNOWLEDGE BASE CONTEXT:
{context}

CUSTOMER QUESTION:
{last_user_message}

Important instructions:
- Use the knowledge base when relevant.
- Do not invent company policies.
- If the answer is not available in the context,
  say that you don't have enough information.
"""

        contents[-1] = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=rag_prompt
                )
            ]
        )


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2
        )
    )


    return response.text