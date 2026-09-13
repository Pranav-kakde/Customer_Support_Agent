from rag.retriever import get_vectorstore



# ==================================================
# RAG SEARCH TOOL
# ==================================================

def search_knowledge_base(
    query: str
) -> dict:
    """
    Search TechNova's knowledge base.

    Use this tool when the customer asks about:
    - Return policy
    - Refund policy
    - Warranty
    - Shipping policy
    - Payment information
    - Troubleshooting
    - Other company policies

    Args:
        query: The customer's question.

    Returns:
        Relevant knowledge-base documents.
    """

    vectorstore = get_vectorstore()


    results = vectorstore.similarity_search_with_score(
        query,
        k=3
    )


    RELEVANCE_THRESHOLD = 1.2


    relevant_documents = []


    for document, score in results:

        if score < RELEVANCE_THRESHOLD:

            relevant_documents.append(
                {
                    "content": document.page_content,
                    "source": document.metadata.get(
                        "source",
                        "Unknown"
                    ),
                    "score": float(score)
                }
            )


    if not relevant_documents:

        return {
            "success": False,
            "message": (
                "No sufficiently relevant information "
                "was found in the TechNova knowledge base."
            )
        }


    return {
        "success": True,
        "results": relevant_documents
    }

