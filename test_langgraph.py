from graph.customer_support_graph import (
    customer_support_graph
)

from langchain_core.messages import HumanMessage


print("\n===================================")
print("   TECHNOVA AI CUSTOMER SUPPORT")
print("===================================\n")


# ============================================================
# CONVERSATION ID
# ============================================================

thread_id = "customer_001"


config = {
    "configurable": {
        "thread_id": thread_id
    }
}


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    question = input("Customer: ")

    if question.lower() in [
        "exit",
        "quit"
    ]:

        print("\nGoodbye!")
        break


    try:

        result = customer_support_graph.invoke(

            {
                "messages": [

                    HumanMessage(
                        content=question
                    )

                ]
            },

            config=config

        )


        final_message = result["messages"][-1]


        print("\n🤖 Agent:")

        print(
            final_message.content
        )

        print()


    except KeyboardInterrupt:

        print(
            "\n\nProgram stopped by user."
        )

        break


    except Exception as error:

        print(
            f"\n❌ Error: {error}"
        )

        print(
            "\nPlease try the question again.\n"
        )

