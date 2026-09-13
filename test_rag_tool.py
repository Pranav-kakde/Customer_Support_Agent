from rag.rag_tool import search_knowledge_base


question = input(
    "Question: "
)


result = search_knowledge_base(
    question
)


print("\n========== RAG RESULT ==========\n")

print(result)