from rag.retriever import get_retriever


retriever = get_retriever()


question = "How long do I have to return a product?"


print("\nSearching knowledge base...\n")


documents = retriever.invoke(question)


print(f"Retrieved {len(documents)} documents\n")


for i, document in enumerate(documents):

    print(f"--- Document {i + 1} ---")

    print(document.page_content)

    print(
        "Source:",
        document.metadata.get("source")
    )

    print()