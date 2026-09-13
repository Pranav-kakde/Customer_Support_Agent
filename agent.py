import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools.support_tools import (
    get_order_status,
    get_customer_details,
    create_support_ticket,
    get_saved_customer_memory,
    save_customer_memory
)

from rag.rag_tool import search_knowledge_base


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-2.5-flash"


# ============================================================
# TOOL 1 — ORDER STATUS
# ============================================================

get_order_status_declaration = types.FunctionDeclaration(
    name="get_order_status",

    description=(
        "Get the current status of a customer order. "
        "Use this when the customer asks about an order, "
        "delivery, shipping, tracking number, or estimated "
        "delivery date."
    ),

    parameters=types.Schema(
        type="OBJECT",

        properties={
            "order_id": types.Schema(
                type="STRING",
                description=(
                    "The customer's order ID, for example TN1001."
                )
            )
        },

        required=["order_id"]
    )
)


# ============================================================
# TOOL 2 — CUSTOMER DETAILS
# ============================================================

get_customer_details_declaration = types.FunctionDeclaration(
    name="get_customer_details",

    description=(
        "Retrieve customer information using the customer ID."
    ),

    parameters=types.Schema(
        type="OBJECT",

        properties={
            "customer_id": types.Schema(
                type="STRING",
                description=(
                    "The customer's ID, for example C001."
                )
            )
        },

        required=["customer_id"]
    )
)


# ============================================================
# TOOL 3 — CREATE SUPPORT TICKET
# ============================================================

create_support_ticket_declaration = types.FunctionDeclaration(
    name="create_support_ticket",

    description=(
        "Create a customer support ticket when the customer "
        "has an issue that requires support assistance."
    ),

    parameters=types.Schema(
        type="OBJECT",

        properties={
            "customer_id": types.Schema(
                type="STRING",
                description="The customer's ID."
            ),

            "issue": types.Schema(
                type="STRING",
                description=(
                    "A clear description of the customer's issue."
                )
            )
        },

        required=[
            "customer_id",
            "issue"
        ]
    )
)


# ============================================================
# TOOL 4 — RAG / KNOWLEDGE BASE
# ============================================================

search_knowledge_base_declaration = types.FunctionDeclaration(
    name="search_knowledge_base",

    description=(
        "Search TechNova's knowledge base for company "
        "policies, FAQs, return policy, refund policy, "
        "warranty information, shipping information, "
        "payment information, and troubleshooting guides."
    ),

    parameters=types.Schema(
        type="OBJECT",

        properties={
            "query": types.Schema(
                type="STRING",
                description=(
                    "The customer's question or search query."
                )
            )
        },

        required=["query"]
    )
)


# ============================================================
# TOOL 5 — GET SAVED CUSTOMER MEMORY
# ============================================================

get_saved_customer_memory_declaration = types.FunctionDeclaration(
    name="get_saved_customer_memory",

    description=(
        "Retrieve long-term saved information about a TechNova "
        "customer, such as communication preferences, product "
        "preferences, or useful customer-support notes. "
        "Use this when the customer asks what you remember "
        "about them or when saved preferences are relevant "
        "to the current request."
    ),

    parameters=types.Schema(
        type="OBJECT",

        properties={
            "customer_id": types.Schema(
                type="STRING",
                description=(
                    "The customer's ID, for example C001."
                )
            )
        },

        required=["customer_id"]
    )
)


# ============================================================
# TOOL 6 — SAVE CUSTOMER MEMORY
# ============================================================

save_customer_memory_declaration = types.FunctionDeclaration(
    name="save_customer_memory",

    description=(
        "Save durable customer information for future "
        "conversations. Use this when the customer explicitly "
        "asks you to remember a preference or important "
        "long-term information. Do not save temporary or "
        "irrelevant conversation details."
    ),

    parameters=types.Schema(
        type="OBJECT",

        properties={
            "customer_id": types.Schema(
                type="STRING",
                description=(
                    "The customer's ID, for example C001."
                )
            ),

            "preferences": types.Schema(
                type="STRING",
                description=(
                    "A durable customer preference, such as "
                    "'prefers email communication'."
                )
            ),

            "notes": types.Schema(
                type="STRING",
                description=(
                    "A useful long-term customer note."
                )
            )
        },

        required=["customer_id"]
    )
)


# ============================================================
# ALL TOOLS
# ============================================================

tools = types.Tool(
    function_declarations=[
        get_order_status_declaration,
        get_customer_details_declaration,
        create_support_ticket_declaration,
        search_knowledge_base_declaration,
        get_saved_customer_memory_declaration,
        save_customer_memory_declaration
    ]
)


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are TechNova's AI Customer Support Agent.

Your job is to help customers with:

- Orders
- Shipping
- Returns
- Refunds
- Warranty
- Payments
- Troubleshooting
- Customer information
- Customer preferences
- Customer memory
- Support tickets


AVAILABLE TOOLS
================

1. search_knowledge_base

Use this tool for:

- Return policy
- Refund policy
- Warranty policy
- Shipping policy
- Payment information
- Troubleshooting
- FAQs
- Company policies

Never invent company policies.


2. get_order_status

Use this tool for:

- Order status
- Delivery
- Shipping
- Tracking number
- Estimated delivery date

Never invent order information.


3. get_customer_details

Use this tool when customer information is required.

Never invent customer information.


4. create_support_ticket

Use this tool when a customer has an issue that requires
support assistance.

Examples:

- Product not working
- Damaged product
- Technical issue
- Delivery problem
- Customer explicitly requests support


5. get_saved_customer_memory

Use this tool when:

- The customer asks what you remember about them.
- The customer asks about their saved preferences.
- A previously saved preference is relevant to the current request.

Only retrieve memory when you have a reliable customer ID.

Never invent customer memory.


6. save_customer_memory

Use this tool when the customer explicitly asks you to:

- Remember something.
- Save a preference.
- Remember how they want to be contacted.
- Remember a useful long-term preference.

Examples:

"My name is Rahul and I prefer email communication.
Please remember this."

"Remember that I prefer email instead of phone calls."

"Please remember that I usually buy headphones."

IMPORTANT:

Only save durable and useful information.

Do NOT save:

- Temporary conversation details
- Random statements
- Passwords
- API keys
- Payment card information
- Sensitive credentials
- Information that the customer did not ask you to remember

Never claim that information has been saved unless the
save_customer_memory tool successfully returns success=True.


CUSTOMER IDENTITY
=================

TechNova's simulated customer database contains:

C001 = Rahul Sharma
C002 = Priya Patil

When the customer clearly identifies themselves by one of
these known names, use the corresponding customer ID in
this simulated portfolio environment.

Rahul Sharma -> C001
Priya Patil -> C002

If the customer's identity is unclear, ask for the customer ID
instead of guessing.

Do not assume that every customer is Rahul Sharma.


MEMORY BEHAVIOR
================

Customer memory is separate from normal conversation history.

Conversation history:
- Used for the current conversation.
- Can contain temporary information.

Customer memory:
- Used for durable preferences and notes.
- Can persist between conversations.

When a customer explicitly asks you to remember something:

1. Identify the customer.
2. Extract only the durable information.
3. Call save_customer_memory.
4. Check the tool result.
5. Tell the customer that it was saved only if the tool
   reports success.

When a customer asks:

"What do you remember about me?"

1. Identify the customer.
2. Call get_saved_customer_memory.
3. Use the returned information.
4. If there is no saved memory, clearly say that no saved
   preferences or notes were found.

Never say that customer memory cannot be stored merely
because this is a simulated environment.


IMPORTANT RULES
================

1. Use tools whenever reliable information is required.

2. Never invent information.

3. Policy questions must use the knowledge base.

4. Order-specific questions must use get_order_status.

5. Customer-specific information must use the appropriate
   customer tool.

6. Customer memory must use the customer-memory tools.

7. You may use multiple tools for one customer question.

8. If a question requires both company policy and
   order-specific information, use both tools.

9. After receiving tool results, reason over the results
   and provide a single clear answer.

10. If information is unavailable, tell the customer.

11. Do not expose internal tool names or implementation
    details to the customer.

12. Keep responses concise and professional.

13. Never claim that an action was completed unless the
    corresponding tool confirms success.

14. Never claim that memory was saved unless the
    save_customer_memory tool confirms success.

15. Never expose customer IDs unless they are useful or
    explicitly requested.
"""


# ============================================================
# GEMINI REQUEST WITH RETRY
# ============================================================

def generate_with_retry(
    contents,
    max_retries=4
):
    """
    Call Gemini with retry handling for temporary errors.

    503 / UNAVAILABLE:
        Retry because these can be temporary.

    429 / RESOURCE_EXHAUSTED:
        Do not repeatedly retry quota errors.
    """

    for attempt in range(max_retries):

        try:

            return client.models.generate_content(

                model=MODEL_NAME,

                contents=contents,

                config=types.GenerateContentConfig(

                    system_instruction=SYSTEM_INSTRUCTION,

                    tools=[tools],

                    temperature=0.2
                )
            )

        except Exception as error:

            error_text = str(error)

            # ------------------------------------------------
            # QUOTA / RATE LIMIT
            # ------------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
            ):
                raise RuntimeError(
                    "Gemini API quota has been exceeded. "
                    "Please try again later."
                )

            # ------------------------------------------------
            # TEMPORARY SERVER ERROR
            # ------------------------------------------------

            is_temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "ServiceUnavailable" in error_text
            )

            if not is_temporary_error:
                raise

            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt

            print(
                "\n⚠️ Gemini temporarily unavailable."
            )

            print(
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


# ============================================================
# EXECUTE TOOL
# ============================================================

def execute_tool(
    function_name,
    function_args
):

    # --------------------------------------------------------
    # ORDER
    # --------------------------------------------------------

    if function_name == "get_order_status":

        return get_order_status(
            function_args["order_id"]
        )


    # --------------------------------------------------------
    # CUSTOMER
    # --------------------------------------------------------

    elif function_name == "get_customer_details":

        return get_customer_details(
            function_args["customer_id"]
        )


    # --------------------------------------------------------
    # SUPPORT TICKET
    # --------------------------------------------------------

    elif function_name == "create_support_ticket":

        return create_support_ticket(
            function_args["customer_id"],
            function_args["issue"]
        )


    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    elif function_name == "search_knowledge_base":

        return search_knowledge_base(
            function_args["query"]
        )


    # --------------------------------------------------------
    # GET CUSTOMER MEMORY
    # --------------------------------------------------------

    elif function_name == "get_saved_customer_memory":

        return get_saved_customer_memory(
            function_args["customer_id"]
        )


    # --------------------------------------------------------
    # SAVE CUSTOMER MEMORY
    # --------------------------------------------------------

    elif function_name == "save_customer_memory":

        return save_customer_memory(
            customer_id=function_args["customer_id"],
            preferences=function_args.get("preferences"),
            notes=function_args.get("notes")
        )


    # --------------------------------------------------------
    # UNKNOWN TOOL
    # --------------------------------------------------------

    return {
        "success": False,
        "message": (
            f"Unknown tool: {function_name}"
        )
    }


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(user_message):

    # --------------------------------------------------------
    # INITIAL CONVERSATION
    # --------------------------------------------------------

    contents = [

        types.Content(

            role="user",

            parts=[
                types.Part.from_text(
                    text=user_message
                )
            ]
        )
    ]


    # --------------------------------------------------------
    # MAXIMUM NUMBER OF TOOL ROUNDS
    # --------------------------------------------------------

    MAX_TOOL_ROUNDS = 5


    for round_number in range(
        MAX_TOOL_ROUNDS
    ):

        print(
            f"\n🧠 Agent round: {round_number + 1}"
        )


        # ----------------------------------------------------
        # ASK GEMINI
        # ----------------------------------------------------

        response = generate_with_retry(
            contents
        )


        # ----------------------------------------------------
        # NO TOOL REQUIRED
        # ----------------------------------------------------

        if not response.function_calls:

            return response.text


        # ----------------------------------------------------
        # ADD GEMINI RESPONSE TO CONVERSATION
        # ----------------------------------------------------

        contents.append(
            response.candidates[0].content
        )


        # ----------------------------------------------------
        # EXECUTE ALL REQUESTED TOOLS
        # ----------------------------------------------------

        tool_response_parts = []


        for function_call in response.function_calls:

            function_name = function_call.name

            function_args = dict(
                function_call.args
            )


            print(
                f"\n🔧 Gemini requested tool: "
                f"{function_name}"
            )

            print(
                f"Arguments: {function_args}"
            )


            # ------------------------------------------------
            # EXECUTE PYTHON TOOL
            # ------------------------------------------------

            result = execute_tool(
                function_name,
                function_args
            )


            print(
                f"Tool result: {result}"
            )


            # ------------------------------------------------
            # CONVERT RESULT INTO GEMINI FUNCTION RESPONSE
            # ------------------------------------------------

            tool_response_parts.append(

                types.Part.from_function_response(

                    name=function_name,

                    response=result
                )
            )


        # ----------------------------------------------------
        # SEND TOOL RESULTS BACK TO GEMINI
        # ----------------------------------------------------

        contents.append(

            types.Content(

                role="tool",

                parts=tool_response_parts
            )
        )


    # --------------------------------------------------------
    # SAFETY LIMIT
    # --------------------------------------------------------

    return (
        "I'm sorry, but I wasn't able to complete "
        "your request after several processing steps. "
        "Please try again."
    )