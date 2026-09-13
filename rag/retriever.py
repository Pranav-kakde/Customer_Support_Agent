import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from langchain_chroma import Chroma


load_dotenv()

CHROMA_PATH = "data/chroma"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


class GeminiEmbeddingFunction:

    def embed_documents(self, texts):

        embeddings = []

        for text in texts:

            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )

            embeddings.append(
                response.embeddings[0].values
            )

        return embeddings


    def embed_query(self, text):

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )

        return response.embeddings[0].values


def get_vectorstore():

    embedding_function = GeminiEmbeddingFunction()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name="technova_support",
        embedding_function=embedding_function
    )

    return vectorstore


def get_retriever():

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )