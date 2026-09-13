import streamlit as st
import uuid
import sqlite3
import os

from langchain_core.messages import HumanMessage

from graph.customer_support_graph import customer_support_graph

from memory.customer_memory import get_customer_memory
from tools.support_tools import create_support_ticket


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Customer Support Agent",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

MEMORY_DIR = "memory"

os.makedirs(
    MEMORY_DIR,
    exist_ok=True
)

CONVERSATION_DB = os.path.join(
    MEMORY_DIR,
    "conversations.db"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    return sqlite3.connect(
        CONVERSATION_DB,
        check_same_thread=False
    )


# ============================================================
# INITIALIZE CONVERSATION DATABASE
# ============================================================

def initialize_database():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (

            thread_id TEXT PRIMARY KEY,

            title TEXT NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            status TEXT DEFAULT 'active',

            rating INTEGER,

            feedback TEXT

        )
        """
    )

    # --------------------------------------------------------
    # Upgrade databases created in earlier phases
    # --------------------------------------------------------

    try:

        cursor.execute(
            "ALTER TABLE conversations "
            "ADD COLUMN status TEXT DEFAULT 'active'"
        )

    except sqlite3.OperationalError:

        pass

    try:

        cursor.execute(
            "ALTER TABLE conversations "
            "ADD COLUMN rating INTEGER"
        )

    except sqlite3.OperationalError:

        pass

    try:

        cursor.execute(
            "ALTER TABLE conversations "
            "ADD COLUMN feedback TEXT"
        )

    except sqlite3.OperationalError:

        pass

    conn.commit()

    conn.close()


initialize_database()


# ============================================================
# CREATE CONVERSATION
# ============================================================

def create_conversation(
    thread_id,
    title="New Conversation"
):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO conversations
        (
            thread_id,
            title
        )
        VALUES (?, ?)
        """,
        (
            thread_id,
            title
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# GET ALL CONVERSATIONS
# ============================================================

def get_conversations():

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            thread_id,
            title,
            created_at,
            updated_at,
            status

        FROM conversations

        ORDER BY updated_at DESC
        LIMIT 3
        """
    )

    conversations = cursor.fetchall()

    conn.close()

    return conversations


# ============================================================
# UPDATE CONVERSATION TITLE
# ============================================================

def update_conversation_title(
    thread_id,
    title
):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE conversations

        SET
            title = ?,
            updated_at = CURRENT_TIMESTAMP

        WHERE thread_id = ?
        """,
        (
            title,
            thread_id
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# UPDATE CONVERSATION TIME
# ============================================================

def update_conversation_time(
    thread_id
):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE conversations

        SET
            updated_at = CURRENT_TIMESTAMP

        WHERE thread_id = ?
        """,
        (
            thread_id,
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# CLOSE CONVERSATION
# ============================================================

def close_conversation(
    thread_id,
    rating=None,
    feedback=None
):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE conversations

        SET
            status = 'closed',
            rating = ?,
            feedback = ?,
            updated_at = CURRENT_TIMESTAMP

        WHERE thread_id = ?
        """,
        (
            rating,
            feedback,
            thread_id,
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# REOPEN CONVERSATION
# ============================================================

def reopen_conversation(
    thread_id
):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE conversations

        SET
            status = 'active',
            updated_at = CURRENT_TIMESTAMP

        WHERE thread_id = ?
        """,
        (
            thread_id,
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# GET CONVERSATION STATUS
# ============================================================

def get_conversation_status(
    thread_id
):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT status

        FROM conversations

        WHERE thread_id = ?
        """,
        (
            thread_id,
        )
    )

    result = cursor.fetchone()

    conn.close()

    if result and result[0]:

        return result[0]

    return "active"


# ============================================================
# DELETE CONVERSATION
# ============================================================

def delete_conversation(
    thread_id
):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM conversations

        WHERE thread_id = ?
        """,
        (
            thread_id,
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# RESTORE LANGGRAPH CONVERSATION
# ============================================================

def load_conversation(
    thread_id
):

    # --------------------------------------------------------
    # Set selected thread
    # --------------------------------------------------------

    st.session_state.thread_id = thread_id

    # --------------------------------------------------------
    # LangGraph configuration
    # --------------------------------------------------------

    config = {

        "configurable": {

            "thread_id": thread_id

        }

    }

    # --------------------------------------------------------
    # Read saved LangGraph state
    # --------------------------------------------------------

    try:

        state = customer_support_graph.get_state(
            config
        )

        saved_messages = state.values.get(
            "messages",
            []
        )

        # ----------------------------------------------------
        # Convert LangChain messages into Streamlit messages
        # ----------------------------------------------------

        restored_messages = []

        for message in saved_messages:

            # ------------------------------------------------
            # HUMAN MESSAGE
            # ------------------------------------------------

            if message.type == "human":

                if message.content:

                    restored_messages.append(
                        {
                            "role": "user",
                            "content": str(
                                message.content
                            )
                        }
                    )

            # ------------------------------------------------
            # AI MESSAGE
            # ------------------------------------------------

            elif message.type == "ai":

                # Ignore tool-call-only AI messages

                if message.content:

                    content = message.content

                    # Normal text response

                    if isinstance(
                        content,
                        str
                    ):

                        restored_messages.append(
                            {
                                "role": "assistant",
                                "content": content
                            }
                        )

        st.session_state.messages = (
            restored_messages
        )

    except Exception as error:

        st.session_state.messages = []

        st.error(
            "⚠️ Could not load conversation."
        )

        st.exception(error)


# ============================================================
# CREATE NEW CONVERSATION
# ============================================================

def new_conversation():

    new_thread_id = str(
        uuid.uuid4()
    )

    st.session_state.thread_id = (
        new_thread_id
    )

    st.session_state.messages = []

    st.session_state.show_feedback = False

    create_conversation(
        new_thread_id,
        "New Conversation"
    )


# ============================================================
# GENERATE SIMPLE TITLE
# ============================================================

def generate_title(
    user_message
):

    title = user_message.strip()

    title = title.replace(
        "\n",
        " "
    )

    if len(title) > 40:

        title = (
            title[:40]
            + "..."
        )

    return title


# ============================================================
# SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(
        uuid.uuid4()
    )

    create_conversation(
        st.session_state.thread_id
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


if "show_feedback" not in st.session_state:

    st.session_state.show_feedback = False


if "pending_input" not in st.session_state:

    st.session_state.pending_input = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🤖 Customer Support"
    )

    st.markdown("---")


    # ========================================================
    # NEW CONVERSATION
    # ========================================================

    if st.button(
        "➕ New Conversation",
        use_container_width=True
    ):

        new_conversation()

        st.rerun()


    st.markdown("---")


    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    st.subheader(
        "💬 Conversations"
    )

    conversations = get_conversations()


    if not conversations:

        st.caption(
            "No conversations yet."
        )


    # --------------------------------------------------------
    # Display conversations
    # --------------------------------------------------------

    for conversation in conversations:

        thread_id = conversation[0]

        title = conversation[1]

        status = (
            conversation[4]
            if len(conversation) > 4
            else "active"
        )

        status_icon = (
            "🟢"
            if status == "active"
            else "⚪"
        )

        if (
            thread_id
            == st.session_state.thread_id
        ):

            button_label = (
                f"{status_icon} {title}"
            )

        else:

            button_label = (
                f"💬 {title}"
            )


        if st.button(
            button_label,
            key=f"conversation_{thread_id}",
            use_container_width=True
        ):

            load_conversation(
                thread_id
            )

            st.rerun()


    st.markdown("---")


    # ========================================================
    # CURRENT CONVERSATION
    # ========================================================

    st.subheader(
        "📌 Current Conversation"
    )

    current_conversation = next(
        (
            conversation
            for conversation in conversations
            if conversation[0]
            == st.session_state.thread_id
        ),
        None
    )


    if current_conversation:

        st.caption(
            current_conversation[1]
        )


    st.caption(
        f"Thread ID:\n"
        f"{st.session_state.thread_id}"
    )


    current_status = (
        get_conversation_status(
            st.session_state.thread_id
        )
    )


    if current_status == "active":

        st.success(
            "● Chat Active"
        )

    else:

        st.info(
            "○ Chat Closed"
        )


    # ========================================================
    # TALK TO CUSTOMER SUPPORT
    # ========================================================

    if st.button(
        "📞 Talk to Customer Support",
        use_container_width=True,
        disabled=current_status == "closed"
    ):

        # Use selected customer ID
        # or default to C001.

        support_customer_id = (
            st.session_state.get(
                "customer_memory_customer_id",
                "C001"
            )
        )

        if not support_customer_id:

            support_customer_id = "C001"

        support_customer_id = (
            support_customer_id
            .strip()
            .upper()
        )

        ticket = create_support_ticket(
            customer_id=support_customer_id,
            issue=(
                "Customer requested human "
                "customer support from the "
                "TechNova AI assistant."
            )
        )

        st.success(
            "Support request created: "
            f"{ticket.get('ticket_id', 'N/A')}"
        )

        st.info(
            "A TechNova support representative "
            "can follow up using the support ticket."
        )


    # ========================================================
    # DELETE CURRENT CONVERSATION
    # ========================================================

    if st.button(
        "🗑️ Delete Current Conversation",
        use_container_width=True
    ):

        delete_conversation(
            st.session_state.thread_id
        )

        new_conversation()

        st.rerun()


    st.markdown("---")


    # ========================================================
    # CAPABILITIES
    # ========================================================

    st.subheader(
        "⚡ Capabilities"
    )

    st.write(
        "📚 RAG Knowledge Base"
    )

    st.write(
        "📦 Order Tracking"
    )

    st.write(
        "👤 Customer Information"
    )

    st.write(
        "🎫 Support Tickets"
    )

    st.write(
        "🧠 Persistent Memory"
    )

    st.write(
        "🔗 LangGraph"
    )

    st.write(
        "✨ Gemini"
    )


# ============================================================
# MAIN UI
# ============================================================

st.title(
    "🤖 AI Customer Support Agent"
)

st.caption(
    "Gemini + RAG + Tool Calling + "
    "LangGraph + SQLite Memory"
)


# ============================================================
# QUICK ACTIONS
# ============================================================

current_status = get_conversation_status(
    st.session_state.thread_id
)

st.markdown(
    "### ⚡ Quick Help"
)


quick_actions = [

    (
        "📦 Track Order",
        "I want to track my order. "
        "My order number is "
    ),

    (
        "↩️ Return / Replace",
        "I want to return or replace "
        "an order. My order number is "
    ),

    (
        "💰 Refund",
        "I want to know the refund status "
        "for my order. My order number is "
    ),

    (
        "❌ Cancel Order",
        "I want to cancel my order. "
        "My order number is "
    ),

    (
        "🛡️ Warranty",
        "I have a warranty question "
        "about my TechNova product."
    ),

    (
        "⚠️ Report a Problem",
        "I want to report a problem "
        "with my TechNova order/product."
    ),

]


cols = st.columns(3)


for index, (
    label,
    prompt
) in enumerate(
    quick_actions
):

    with cols[index % 3]:

        if st.button(
            label,
            key=f"quick_action_{index}",
            use_container_width=True,
            disabled=current_status == "closed"
        ):

            st.session_state.pending_input = (
                prompt
            )

            st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

pending_input = (
    st.session_state.pop(
        "pending_input",
        None
    )
)


user_input = st.chat_input(

    "Chat is closed. Start a new "
    "conversation from the sidebar."

    if current_status == "closed"

    else

    "Ask me about your order, "
    "refund, warranty, shipping..."
)


if (
    pending_input
    and current_status == "active"
):

    user_input = pending_input


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if (
    user_input
    and current_status == "active"
):

    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )


    # ========================================================
    # GENERATE TITLE
    # ========================================================

    conversations = get_conversations()


    current_conversation = next(
        (
            conversation
            for conversation in conversations
            if conversation[0]
            == st.session_state.thread_id
        ),
        None
    )


    if current_conversation:

        current_title = (
            current_conversation[1]
        )


        if (
            current_title
            == "New Conversation"
        ):

            title = generate_title(
                user_input
            )

            update_conversation_title(
                st.session_state.thread_id,
                title
            )


    # ========================================================
    # LANGGRAPH CONFIG
    # ========================================================

    config = {

        "configurable": {

            "thread_id":
                st.session_state.thread_id

        }

    }


    # ========================================================
    # INVOKE CUSTOMER SUPPORT AGENT
    # ========================================================

    try:

        result = customer_support_graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ]
            },
            config=config
        )


    except RuntimeError as error:

        error_text = str(
            error
        ).lower()


        # ----------------------------------------------------
        # GEMINI QUOTA ERROR
        # ----------------------------------------------------

        if (
            "quota" in error_text
            or "gemini api" in error_text
            or "resource_exhausted" in error_text
        ):

            st.warning(
                "⚠️ Our AI assistant is "
                "temporarily unavailable. "
                "Please try again in a little while."
            )

            st.info(
                "If you need immediate assistance, "
                "please use **Talk to Customer Support**."
            )


        # ----------------------------------------------------
        # OTHER RUNTIME ERROR
        # ----------------------------------------------------

        else:

            st.error(
                "⚠️ Sorry, I couldn't process "
                "your request right now. "
                "Please try again."
            )

            # Show technical error only while
            # developing locally.

            st.exception(error)


        st.stop()


    except Exception as error:

        st.error(
            "⚠️ Sorry, something went wrong "
            "while processing your request."
        )

        # IMPORTANT:
        # Keep this during development so that
        # we can see the real error.
        #
        # Remove this before final deployment
        # if you don't want customers to see it.

        st.exception(error)

        st.info(
            "If the problem continues, "
            "please use **Talk to Customer Support**."
        )

        st.stop()


    # ========================================================
    # UPDATE CONVERSATION TIMESTAMP
    # ========================================================

    update_conversation_time(
        st.session_state.thread_id
    )


    # ========================================================
    # GET LAST AI RESPONSE
    # ========================================================

    ai_response = None


    for message in reversed(
        result.get(
            "messages",
            []
        )
    ):

        if message.type == "ai":

            if message.content:

                # --------------------------------------------
                # Normal string response
                # --------------------------------------------

                if isinstance(
                    message.content,
                    str
                ):

                    ai_response = (
                        message.content
                    )


                # --------------------------------------------
                # Structured response
                # --------------------------------------------

                elif isinstance(
                    message.content,
                    list
                ):

                    text_parts = []


                    for part in (
                        message.content
                    ):

                        if isinstance(
                            part,
                            dict
                        ):

                            text = part.get(
                                "text"
                            )

                            if text:

                                text_parts.append(
                                    text
                                )


                    if text_parts:

                        ai_response = (
                            "\n".join(
                                text_parts
                            )
                        )


                if ai_response:

                    break


    # ========================================================
    # FALLBACK RESPONSE
    # ========================================================

    if not ai_response:

        ai_response = (
            "I'm sorry, I couldn't "
            "generate a response right now. "
            "Please try again."
        )


    # ========================================================
    # DISPLAY AI RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            ai_response
        )


    # ========================================================
    # SAVE AI RESPONSE TO UI HISTORY
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )


# ============================================================
# CUSTOMER MEMORY
# ============================================================

with st.sidebar:

    st.markdown("---")

    st.subheader(
        "👤 Customer Memory"
    )


    customer_id = st.text_input(
        "Customer ID",
        value="C001",
        key="customer_memory_customer_id"
    )


    if st.button(
        "🔍 Load Customer Memory",
        use_container_width=True,
        key="load_customer_memory"
    ):

        customer_id = (
            customer_id
            .strip()
            .upper()
        )


        if not customer_id:

            st.warning(
                "Please enter a Customer ID."
            )


        else:

            customer_memory = (
                get_customer_memory(
                    customer_id
                )
            )


            if customer_memory:

                st.success(
                    "Customer memory loaded."
                )

                st.json(
                    customer_memory
                )


            else:

                st.info(
                    f"No saved memory found "
                    f"for {customer_id}."
                )


# ============================================================
# END CHAT / FEEDBACK
# ============================================================

st.markdown("---")


final_status = get_conversation_status(
    st.session_state.thread_id
)


# ============================================================
# END CHAT BUTTON
# ============================================================

if final_status == "active":

    st.subheader(
        "🔚 End this conversation"
    )

    st.caption(
        "Your conversation will be saved. "
        "You can start a new conversation anytime."
    )


    if st.button(
        "🔴 End Chat",
        use_container_width=True
    ):

        st.session_state.show_feedback = (
            True
        )

        st.rerun()


# ============================================================
# FEEDBACK FORM
# ============================================================

if st.session_state.get(
    "show_feedback",
    False
):

    st.markdown(
        "### ⭐ How was your TechNova support experience?"
    )


    rating = st.radio(
        "Rate this conversation",
        [1, 2, 3, 4, 5],
        horizontal=True,
        index=4
    )


    feedback = st.text_area(
        "Optional feedback",
        placeholder=(
            "Tell us how we can improve..."
        )
    )


    if st.button(
        "Submit & Close Chat",
        type="primary",
        use_container_width=True
    ):

        close_conversation(
            st.session_state.thread_id,
            rating,
            feedback.strip()
            if feedback.strip()
            else None
        )


        st.session_state.show_feedback = (
            False
        )


        st.success(
            "Chat closed successfully. "
            "Your conversation history has been saved."
        )


        st.rerun()