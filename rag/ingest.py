import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------

KNOWLEDGE_BASE_PATH = "data/knowledge_base"
CHROMA_PATH = "data/chroma"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from your .env file."
    )


# Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# --------------------------------------------------
# Custom Gemini Embedding Function
# --------------------------------------------------

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


# --------------------------------------------------
# Load documents
# --------------------------------------------------

def load_documents():

    documents = []

    for file_path in Path(
        KNOWLEDGE_BASE_PATH
    ).glob("*.txt"):

        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        documents.extend(
            loader.load()
        )

    return documents


# --------------------------------------------------
# Create Vector Database
# --------------------------------------------------

def create_vector_database():

    print("\nLoading documents...")

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents."
    )


    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )


    # Gemini embedding function
    embedding_function = GeminiEmbeddingFunction()


    # Create ChromaDB
    print("\nCreating ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH,
        collection_name="technova_support"
    )


    print(
        "\n✅ Vector database created successfully!"
    )

    print(
        f"Stored at: {CHROMA_PATH}"
    )


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    create_vector_database()