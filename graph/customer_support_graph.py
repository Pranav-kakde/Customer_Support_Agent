from typing import TypedDict, Annotated, Optional
import os
import time
import sqlite3

from dotenv import load_dotenv

from google import genai
from google.genai import types

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)

from tools.support_tools import (
    get_order_status,
    get_customer_details,
    create_support_ticket,
    cancel_order,
    request_return,
    request_replacement,
    check_refund_status
)

from rag.rag_tool import search_knowledge_base


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"


# ============================================================
# 2. SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are TechNova's AI Customer Support Agent.

TechNova is an e-commerce company that sells consumer electronics.

Your job is to help customers with:

- Order tracking
- Order cancellation
- Returns
- Replacements
- Refunds
- Warranty
- Shipping
- Product problems
- Customer information
- Support tickets
- General TechNova policies

You must behave like a professional e-commerce customer support agent.

==================================================
IMPORTANT RULES
==================================================

1. NEVER invent information.

If you do not have the information, say that you do not have it.

--------------------------------------------------

2. COMPANY POLICY QUESTIONS

For questions involving:

- Return policy
- Refund policy
- Warranty
- Shipping policy
- Troubleshooting
- Other TechNova policies

use the search_knowledge_base tool.

Do not invent company policies.

--------------------------------------------------

3. ORDER-SPECIFIC QUESTIONS

For questions involving a specific order:

- Use get_order_status.

Examples:

"Where is my order TN1001?"

"Has my order shipped?"

"When will TN1001 arrive?"

--------------------------------------------------

4. CUSTOMER INFORMATION

Use get_customer_details when customer information is required.

--------------------------------------------------

5. ORDER CANCELLATION

If the customer wants to cancel an order:

Step 1:
Check the order using get_order_status.

Step 2:
If necessary, check the relevant TechNova cancellation policy using
search_knowledge_base.

Step 3:
If the order is eligible, use cancel_order.

Do not claim an order was cancelled unless cancel_order successfully confirms it.

--------------------------------------------------

6. RETURN REQUEST

If the customer wants to return an order:

Step 1:
Check the order using get_order_status.

Step 2:
Check the TechNova return policy using search_knowledge_base.

Step 3:
Determine whether the order appears eligible.

Step 4:
If appropriate, use request_return.

Do not claim that a return was created unless the tool confirms it.

--------------------------------------------------

7. REPLACEMENT REQUEST

If the customer wants a replacement:

Step 1:
Check the order.

Step 2:
Check warranty/return/replacement policy when necessary.

Step 3:
Determine eligibility.

Step 4:
Use request_replacement when appropriate.

Do not claim a replacement was created unless the tool confirms it.

--------------------------------------------------

8. REFUND STATUS

If the customer asks about a refund:

Use check_refund_status for an order-specific refund status.

If the question concerns refund policy, also use
search_knowledge_base.

--------------------------------------------------

9. CUSTOMER PROBLEMS

Examples:

"My headphones aren't working."

"My product arrived damaged."

"My watch stopped working."

First understand the issue.

Then use the appropriate policy/tool.

Do NOT immediately create a support ticket.

Try to solve the problem using:

- Knowledge base
- Order information
- Customer information
- Available business tools

--------------------------------------------------

10. SUPPORT TICKETS

Do NOT create a support ticket for every question.

Create a ticket only when:

- The issue cannot be resolved by the available tools
- The customer requires human intervention
- A complex complaint requires manual investigation
- A technical/problem case cannot be resolved through available guidance
- A business action fails and needs human assistance

If the AI can solve the issue, solve it instead of creating a ticket.

--------------------------------------------------

11. HUMAN SUPPORT

If the issue genuinely requires human assistance:

Create a support ticket when appropriate.

Then tell the customer that their issue has been escalated.

Provide the ticket ID returned by the tool.

Never claim that you connected the customer to a real human agent unless
a real human-support integration exists.

--------------------------------------------------

12. DEMO SYSTEM

The TechNova tools are currently a simulated/demo e-commerce backend.

Never claim that you accessed a real external company's database,
payment gateway, shipping company, or live customer-service system.

You may say that the action was processed in the TechNova system.

--------------------------------------------------

13. MULTI-STEP QUESTIONS

A customer may ask a question requiring multiple tools.

Example:

"Can I return TN1001?"

You may need:

get_order_status
+
search_knowledge_base

Combine the results before responding.

--------------------------------------------------

14. CONVERSATION CONTEXT

Use previous conversation messages.

If the customer previously mentioned an order ID, customer ID,
or problem, use that context when appropriate.

Do not repeatedly ask for information that is already available.

--------------------------------------------------

15. CUSTOMER EXPERIENCE

Be:

- Friendly
- Professional
- Clear
- Concise
- Helpful

Do not expose internal tool names.

Do not explain internal agent architecture to customers.

--------------------------------------------------

16. WHEN AN ACTION IS NOT POSSIBLE

Clearly explain why.

Example:

"I checked the order and it is already being processed for shipment,
so cancellation is no longer available."

Then suggest the next useful option.

--------------------------------------------------

17. CONFIRMATION BEFORE BUSINESS ACTIONS

The following actions change an order or create a customer request:
- cancel_order
- request_return
- request_replacement

Before any of these actions is executed, the customer MUST explicitly confirm.

Required workflow:
1. Check the order and relevant policy first.
2. Determine whether the requested action appears eligible.
3. Explain what will happen and identify the order.
4. Ask for explicit confirmation, such as "Yes, proceed" or "Confirm".
5. Only after explicit confirmation execute the action.
6. If the customer declines, do not execute the action.

Never treat a vague statement as confirmation. If the customer has not clearly
confirmed, ask again. Never execute cancel/return/replacement merely because
the customer originally requested it.

Do not ask for confirmation for read-only actions such as order tracking,
policy lookup, customer details, or refund-status checks.

--------------------------------------------------

18. INTELLIGENT HUMAN ESCALATION

Do not escalate a customer to human support simply because the question is
difficult. First attempt to solve the issue using the knowledge base, order
information, customer information, and available business tools.

Escalate when one or more of these conditions is true:
- The customer explicitly asks to speak with customer support.
- The available tools cannot complete the requested business action.
- The issue requires manual investigation that the demo system cannot perform.
- The customer reports a serious unresolved product/problem case after available troubleshooting or policy guidance.
- There is conflicting, missing, or insufficient information to safely resolve the case.
- A transaction/action fails and cannot be safely completed by the available tools.
- The customer remains dissatisfied after a reasonable resolution attempt and requests human assistance.

Before escalating:
1. Summarize the customer's issue internally from the conversation.
2. Gather any known customer ID, order ID, and relevant details already present.
3. Do not ask again for information already available.
4. If the issue can still be solved safely, solve it instead of escalating.

When escalation is necessary:
1. Use create_support_ticket with the known customer ID and a concise, useful description of the issue.
2. Do not create multiple duplicate tickets for the same unresolved issue during the same conversation unless a new issue clearly requires it.
3. Tell the customer that the issue has been escalated and provide the ticket ID.
4. Clearly explain the next available support option.
5. Never claim that a real human is currently on the call/chat unless a real human-support integration exists.

If the customer asks for a voice call, explain that this demo can initiate or record a support request, but it does not contain a real-time voice connection. Do not pretend that a call has been connected.

--------------------------------------------------

19. ESCALATION PRIORITY

Use this priority order:

LEVEL 1 - SELF SERVICE
Solve using RAG, order/customer lookup, and available actions.

LEVEL 2 - GUIDED RESOLUTION
Ask one focused clarification or provide troubleshooting/next steps when the issue needs more information but can still be solved by the AI.

LEVEL 3 - HUMAN ESCALATION
Create a support ticket when the AI cannot safely or completely resolve the issue.

Never skip directly to Level 3 when Level 1 or Level 2 can reasonably solve the customer's problem.

--------------------------------------------------

20. FINAL RESPONSE

After tools have returned their results:

- Understand the result
- Give the customer a clear answer
- Mention important next steps
- Mention ticket/action IDs when applicable

Do not simply dump raw tool results.
"""


# ============================================================
# 3. STATE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_action: Optional[dict]
    action_confirmed: bool


# ============================================================
# 4. GEMINI TOOL DECLARATIONS
# ============================================================

tools = [

    # --------------------------------------------------------
    # ORDER STATUS
    # --------------------------------------------------------

    {
        "function_declarations": [
            {
                "name": "get_order_status",
                "description": (
                    "Get the current status and details of a TechNova "
                    "customer order."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "order_id": {
                            "type": "STRING",
                            "description": "TechNova order ID, for example TN1001"
                        }
                    },
                    "required": ["order_id"]
                }
            }
        ]
    },

    # --------------------------------------------------------
    # CUSTOMER DETAILS
    # --------------------------------------------------------

    {
        "function_declarations": [
            {
                "name": "get_customer_details",
                "description": (
                    "Get customer information from the TechNova customer "
                    "system."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "customer_id": {
                            "type": "STRING",
                            "description": "TechNova customer ID, for example C001"
                        }
                    },
                    "required": ["customer_id"]
                }
            }
        ]
    },

    # --------------------------------------------------------
    # KNOWLEDGE BASE / RAG
    # --------------------------------------------------------

    {
        "function_declarations": [
            {
                "name": "search_knowledge_base",
                "description": (
                    "Search TechNova's knowledge base for company policies, "
                    "FAQs, shipping, returns, refunds, warranty and "
                    "troubleshooting information."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "Question or information to search for"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    },

    # --------------------------------------------------------
    # CREATE SUPPORT TICKET
    # --------------------------------------------------------

    {
        "function_declarations": [
            {
                "name": "create_support_ticket",
                "description": (
                    "Create a TechNova support ticket when an issue "
                    "requires human assistance."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "customer_id": {
                            "type": "STRING",
                            "description": "TechNova customer ID"
                        },
                        "issue": {
                            "type": "STRING",
                            "description": "Description of the customer's issue"
                        }
                    },
                    "required": [
                        "customer_id",
                        "issue"
                    ]
                }
            }
        ]
    },

    # --------------------------------------------------------
    # CANCEL ORDER
    # --------------------------------------------------------

    {
        "function_declarations": [
            {
                "name": "cancel_order",
                "description": (
                    "Cancel a TechNova order when the order is eligible "
                    "for cancellation."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "order_id": {
                            "type": "STRING",
                            "description": "TechNova order ID"
                        }
                    },
                    "required": ["order_id"]
                }
            }
        ]
    },

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    {
        "function_declarations": [
            {
                "name": "request_return",
                "description": (
                    "Create a return request for an eligible TechNova order."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "order_id": {
                            "type": "STRING",
                            "description": "TechNova order ID"
                        },
                        "reason": {
                            "type": "STRING",
                            "description": "Reason for requesting the return"
                        }
                    },
                    "required": [
                        "order_id",
                        "reason"
                    ]
                }
            }
        ]
    },

    # --------------------------------------------------------
    # REPLACEMENT
    # --------------------------------------------------------

    {
        "function_declarations": [
            {
                "name": "request_replacement",
                "description": (
                    "Create a replacement request for an eligible "
                    "TechNova order."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "order_id": {
                            "type": "STRING",
                            "description": "TechNova order ID"
                        },
                        "reason": {
                            "type": "STRING",
                            "description": "Reason for requesting replacement"
                        }
                    },
                    "required": [
                        "order_id",
                        "reason"
                    ]
                }
            }
        ]
    },

    # --------------------------------------------------------
    # REFUND STATUS
    # --------------------------------------------------------

    {
        "function_declarations": [
            {
                "name": "check_refund_status",
                "description": (
                    "Check the refund status for a TechNova order."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "order_id": {
                            "type": "STRING",
                            "description": "TechNova order ID"
                        }
                    },
                    "required": ["order_id"]
                }
            }
        ]
    }
]


# ============================================================
# 5. CONVERT LANGCHAIN MESSAGES → GEMINI
# ============================================================

def convert_messages_to_gemini(messages):

    contents = []

    for message in messages:

        # ----------------------------------------------------
        # USER
        # ----------------------------------------------------

        if isinstance(message, HumanMessage):

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=str(message.content)
                        )
                    ]
                )
            )

        # ----------------------------------------------------
        # TOOL
        # ----------------------------------------------------

        elif isinstance(message, ToolMessage):

            contents.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=message.name,
                            response={
                                "result": message.content
                            }
                        )
                    ]
                )
            )

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        elif isinstance(message, AIMessage):

            if message.content:

                contents.append(
                    types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(
                                text=str(message.content)
                            )
                        ]
                    )
                )

    return contents


# ============================================================
# 6. GEMINI CALL
# ============================================================

def call_gemini(contents):

    max_retries = 4

    for attempt in range(max_retries):

        try:

            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=tools,
                    temperature=0.2
                )
            )

        except Exception as error:

            error_text = str(error)

            # ------------------------------------------------
            # IMPORTANT:
            # Do NOT retry Gemini free-tier quota exhaustion.
            # ------------------------------------------------

            quota_error = (
                "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
                or "free_tier" in error_text.lower()
            )

            if quota_error:

                raise RuntimeError(
                    "Gemini API quota has been exhausted. "
                    "Please wait and try again later."
                )

            temporary = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "RemoteProtocolError" in error_text
            )

            if not temporary:
                raise

            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt

            print(
                f"\n⚠️ Gemini temporary error. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


# ============================================================
# 7. CHATBOT NODE
# ============================================================

def chatbot_node(state: AgentState):

    messages = state["messages"]
    pending_action = state.get("pending_action")

    # --------------------------------------------------------
    # EXPLICIT CONFIRMATION HANDLING
    # --------------------------------------------------------
    # Handle a clear yes/no response in code while an action is
    # waiting for confirmation. This makes the confirmation gate
    # deterministic instead of relying only on the LLM.
    if pending_action and messages and isinstance(messages[-1], HumanMessage):
        user_text = str(messages[-1].content).strip().lower()
        normalized = user_text.rstrip(".!?")

        yes_phrases = {
            "yes",
            "yes please",
            "yes, proceed",
            "proceed",
            "confirm",
            "confirmed",
            "do it",
            "go ahead",
            "please proceed",
            "continue",
        }

        no_phrases = {
            "no",
            "no thanks",
            "no thank you",
            "cancel",
            "don't",
            "do not",
            "stop",
        }

        if normalized in yes_phrases:
            action_name = pending_action["name"]
            action_args = pending_action["args"]

            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": action_name,
                                "args": action_args,
                                "id": f"confirm_{action_name}_0"
                            }
                        ]
                    )
                ],
                "action_confirmed": True,
            }

        if normalized in no_phrases:
            return {
                "messages": [
                    AIMessage(
                        content="No problem. I won't make that change to your order."
                    )
                ],
                "pending_action": None,
                "action_confirmed": False,
            }

    contents = convert_messages_to_gemini(messages)

    response = call_gemini(contents)

    # --------------------------------------------------------
    # GEMINI WANTS TO CALL TOOLS
    # --------------------------------------------------------

    if response.function_calls:
        tool_calls = []

        for function_call in response.function_calls:
            tool_calls.append(
                {
                    "name": function_call.name,
                    "args": function_call.args or {},
                    "id": f"call_{function_call.name}_{len(tool_calls)}"
                }
            )

        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=tool_calls
                )
            ]
        }

    # --------------------------------------------------------
    # NORMAL FINAL RESPONSE
    # --------------------------------------------------------

    final_text = response.text

    if not final_text:
        final_text = (
            "I'm sorry, but I couldn't generate a response right now. "
            "Please try again."
        )

    return {
        "messages": [
            AIMessage(
                content=final_text
            )
        ]
    }


# ============================================================
# 8. TOOLS NODE
# ============================================================

def tools_node(state: AgentState):

    last_message = state["messages"][-1]

    results = []
    pending_action = state.get("pending_action")
    action_confirmed = state.get("action_confirmed", False)

    confirmation_required = {
        "cancel_order",
        "request_return",
        "request_replacement",
    }

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        # ----------------------------------------------------
        # CONFIRMATION GATE
        # ----------------------------------------------------
        if tool_name in confirmation_required and not action_confirmed:
            pending_action = {
                "name": tool_name,
                "args": tool_args,
            }

            result = {
                "success": False,
                "confirmation_required": True,
                "action": tool_name,
                "message": (
                    "The customer must explicitly confirm this action before "
                    "it can be executed. Ask for confirmation and do not execute "
                    "the action yet."
                )
            }

            results.append(
                ToolMessage(
                    content=str(result),
                    name=tool_name,
                    tool_call_id=tool_id
                )
            )
            continue

        # ----------------------------------------------------
        # ORDER STATUS
        # ----------------------------------------------------
        if tool_name == "get_order_status":
            result = get_order_status(
                tool_args["order_id"]
            )

        # ----------------------------------------------------
        # CUSTOMER DETAILS
        # ----------------------------------------------------
        elif tool_name == "get_customer_details":
            result = get_customer_details(
                tool_args["customer_id"]
            )

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------
        elif tool_name == "search_knowledge_base":
            result = search_knowledge_base(
                tool_args["query"]
            )

        # ----------------------------------------------------
        # SUPPORT TICKET
        # ----------------------------------------------------
        elif tool_name == "create_support_ticket":
            result = create_support_ticket(
                tool_args["customer_id"],
                tool_args["issue"]
            )

        # ----------------------------------------------------
        # CANCEL ORDER
        # ----------------------------------------------------
        elif tool_name == "cancel_order":
            result = cancel_order(
                tool_args["order_id"]
            )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------
        elif tool_name == "request_return":
            result = request_return(
                tool_args["order_id"],
                tool_args["reason"]
            )

        # ----------------------------------------------------
        # REPLACEMENT
        # ----------------------------------------------------
        elif tool_name == "request_replacement":
            result = request_replacement(
                tool_args["order_id"],
                tool_args["reason"]
            )

        # ----------------------------------------------------
        # REFUND
        # ----------------------------------------------------
        elif tool_name == "check_refund_status":
            result = check_refund_status(
                tool_args["order_id"]
            )

        # ----------------------------------------------------
        # UNKNOWN TOOL
        # ----------------------------------------------------
        else:
            result = {
                "success": False,
                "message": f"Unknown tool: {tool_name}"
            }

        results.append(
            ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_id
            )
        )

    # A confirmed action has now been passed to the actual business
    # tool, so clear the one-time confirmation gate.
    if action_confirmed:
        pending_action = None
        action_confirmed = False

    return {
        "messages": results,
        "pending_action": pending_action,
        "action_confirmed": action_confirmed,
    }


# ============================================================
# 9. ROUTER
# ============================================================

def route_after_chatbot(state: AgentState):

    last_message = state["messages"][-1]

    if (
        isinstance(last_message, AIMessage)
        and last_message.tool_calls
    ):
        return "tools"

    return END


# ============================================================
# 10. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(AgentState)

builder.add_node(
    "chatbot",
    chatbot_node
)

builder.add_node(
    "tools",
    tools_node
)

builder.add_edge(
    START,
    "chatbot"
)

builder.add_conditional_edges(
    "chatbot",
    route_after_chatbot
)

builder.add_edge(
    "tools",
    "chatbot"
)


# ============================================================
# 11. SQLITE PERSISTENT MEMORY
# ============================================================

MEMORY_DIR = "memory"

os.makedirs(
    MEMORY_DIR,
    exist_ok=True
)

DATABASE_PATH = os.path.join(
    MEMORY_DIR,
    "checkpoints.db"
)

sqlite_connection = sqlite3.connect(
    DATABASE_PATH,
    check_same_thread=False
)

memory = SqliteSaver(
    sqlite_connection
)


# ============================================================
# 12. COMPILE GRAPH
# ============================================================

customer_support_graph = builder.compile(
    checkpointer=memory
)