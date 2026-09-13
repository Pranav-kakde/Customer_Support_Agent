# 🤖 AI Customer Support Agent

An intelligent **AI-powered customer support agent** built with **Google Gemini, LangGraph, RAG, tool calling, and persistent memory**.

The agent is designed for an e-commerce support environment where customers can ask questions about orders, returns, refunds, shipping, warranty, and troubleshooting. It can retrieve relevant information from a knowledge base, use tools to check customer/order/ticket information, and maintain conversation context across interactions.

---

## 🚀 Features

* 🧠 **Google Gemini LLM** for natural-language understanding and response generation
* 🔄 **LangGraph** for building a structured, stateful agent workflow
* 🔎 **RAG (Retrieval-Augmented Generation)** for knowledge-base-based answers
* 🛠️ **Tool Calling** for accessing customer, order, and ticket information
* 💾 **SQLite Persistent Memory** for maintaining conversation history
* 💬 **Streamlit UI** for an interactive customer-support chat interface
* 🛡️ **Error Handling** for safer tool and agent execution
* 🧪 **Automated Tests** for agent, RAG, tools, embeddings, and LangGraph components
* 🐳 **Docker Support** for containerized deployment
* 📚 **E-commerce Knowledge Base** covering returns, refunds, shipping, payments, warranty, and troubleshooting

---

## 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │     User / Customer │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Streamlit App    │
                         │       app.py        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    LangGraph Agent  │
                         │ customer_support_   │
                         │       graph.py      │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
          ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
          │ Gemini LLM   │  │ RAG System    │  │ Tool Calling │
          └──────────────┘  └──────┬───────┘  └──────┬───────┘
                                   │                   │
                                   ▼                   ▼
                           ┌──────────────┐    ┌──────────────┐
                           │ Knowledge    │    │ Customer /   │
                           │ Base        │    │ Order /      │
                           │             │    │ Ticket Data  │
                           └──────────────┘    └──────────────┘
                                   
                                    ▼
                         ┌─────────────────────┐
                         │ Persistent Memory   │
                         │      SQLite         │
                         └─────────────────────┘
```

---

## 🧰 Tech Stack

| Technology    | Purpose                             |
| ------------- | ----------------------------------- |
| Python        | Core programming language           |
| Google Gemini | Large Language Model                |
| LangGraph     | Agent workflow and state management |
| RAG           | Knowledge retrieval                 |
| Streamlit     | User interface                      |
| SQLite        | Persistent conversation memory      |
| Docker        | Containerization                    |
| JSON          | Customer, order and ticket data     |
| Git/GitHub    | Version control                     |

---

## 📁 Project Structure

```text
Customer_Support_Agent/
│
├── app.py
├── agent.py
├── llm.py
├── prompts.py
├── requirements.txt
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── README.md
│
├── data/
│   ├── database/
│   │   ├── customers.json
│   │   ├── orders.json
│   │   └── tickets.json
│   │
│   └── knowledge_base/
│       ├── payment_faq.txt
│       ├── refund_policy.txt
│       ├── return_policy.txt
│       ├── shipping_policy.txt
│       ├── troubleshooting.txt
│       └── warranty_policy.txt
│
├── graph/
│   ├── __init__.py
│   └── customer_support_graph.py
│
├── rag/
│   ├── ingest.py
│   ├── rag_tool.py
│   └── retriever.py
│
├── tools/
│   └── support_tools.py
│
├── utils/
│   ├── __init__.py
│   └── error_handler.py
│
└── tests/
    ├── test_agent.py
    ├── test_embedding.py
    ├── test_langgraph.py
    ├── test_phase9.py
    ├── test_rag.py
    ├── test_rag_tool.py
    └── test_tools.py
```

> Test files are currently located in the project root in this repository.

---

## 🔧 Agent Capabilities

### 1. Order Status

The agent can use order information to answer questions such as:

```text
Where is my order?
Can you check my order status?
What is the tracking number for my order?
```

---

### 2. Customer Lookup

The agent can retrieve customer information when required for support workflows.

Example:

```text
Can you check my customer details?
```

---

### 3. Ticket Lookup

The agent can retrieve support-ticket information.

Example:

```text
What is the status of my support ticket?
```

---

### 4. Knowledge Base Search

The RAG system retrieves relevant information from the support knowledge base.

Supported topics include:

* Returns
* Refunds
* Shipping
* Payments
* Warranty
* Troubleshooting

Example:

```text
What is your refund policy?
How long do I have to return a product?
Does this product have a warranty?
```

---

## 🧠 RAG Pipeline

The Retrieval-Augmented Generation pipeline works approximately as follows:

```text
User Question
      │
      ▼
Question Processing
      │
      ▼
Knowledge Base Retrieval
      │
      ▼
Relevant Documents
      │
      ▼
Gemini
      │
      ▼
Context-Aware Answer
```

Instead of relying only on the LLM's internal knowledge, the agent retrieves relevant information from the project's support documents before generating an answer.

---

## 🛠️ Tool Calling

The agent can select tools when external application data is required.

Current support tools include:

```text
get_order_status
customer lookup
ticket lookup
search_knowledge_base
```

This allows the LLM to decide when it needs additional information instead of attempting to answer every question directly.

---

## 💾 Persistent Memory

The application uses SQLite-based persistence to maintain conversation information.

This allows the support agent to maintain context across conversations instead of treating every interaction as completely independent.

Local database files are intentionally excluded from Git using `.gitignore`.

---

## 🖥️ Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Customer_Support_Agent.git
cd Customer_Support_Agent
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

**Never commit your `.env` file or API key to GitHub.**

---

### 5. Run the application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 🐳 Run with Docker

Build and start the application using:

```bash
docker compose up -d --build
```

Check running containers:

```bash
docker compose ps
```

To stop the application:

```bash
docker compose down
```

---

## 🧪 Running Tests

Run the test suite using:

```bash
pytest
```

Individual test files can also be executed, for example:

```bash
pytest test_rag.py
```

```bash
pytest test_tools.py
```

```bash
pytest test_langgraph.py
```

---

## 💬 Example Questions

Try asking the agent:

```text
What is your refund policy?

How long does shipping normally take?

Can I return my product?

What is the warranty policy?

Can you check my order status?

Can you find my support ticket?

My product is not working. What should I do?

What payment methods are supported?
```

You can also test edge cases such as:

```text
I want to cancel my order.

Check an invalid order ID.

Check an invalid customer ID.

Check an invalid ticket ID.

Ask a question unrelated to the knowledge base.

Ask the same question multiple times.
```

---

## 🔐 Security

The following files and directories are excluded from version control:

```text
.env
venv/
.venv/
__pycache__/
memory/*.db
memory/*.db-shm
memory/*.db-wal
memory/*.sqlite
memory/*.sqlite3
```

API credentials and local persistent databases should never be committed to the repository.

---

## 🎯 Project Goals

This project was built to demonstrate practical implementation of modern AI application development concepts:

* Large Language Models
* Retrieval-Augmented Generation
* AI Agents
* Tool Calling
* Agentic Workflows
* LangGraph
* Persistent Memory
* Streamlit Applications
* Docker Containerization
* Automated Testing

The focus is on building an end-to-end AI support system rather than a simple LLM chatbot.

---

## 🔮 Future Improvements

Potential future improvements include:

* 🎙️ Voice-based customer support
* 📞 Human-agent escalation
* 🌐 REST API using FastAPI
* 🔐 User authentication
* 📊 Support analytics dashboard
* 🧾 Automatic ticket creation
* 📧 Email support integration
* 💬 WhatsApp integration
* ☁️ Cloud deployment
* 📈 Monitoring and logging
* 🧠 Improved conversation memory
* 🔄 More advanced agent routing

---

## 👨‍💻 Author

**Pranav Kakde**

AI/ML & GenAI Enthusiast

Interested in:

* Artificial Intelligence
* Machine Learning
* Generative AI
* AI Agents
* Data Analytics
* Data Engineering

---

## ⭐ Acknowledgment

This project was developed as a hands-on implementation of modern AI agent architecture, combining **LLMs, RAG, tool calling, LangGraph, memory, and application deployment** into a single customer-support system.

If you find this project useful, consider giving the repository a ⭐.
