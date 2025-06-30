# 📄 Chat with PDF — RAG-based Document QA using Gemini + Pinecone + Flask

This project is a **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents and ask questions about them. The system uses:

- 🧠 **Google Gemini** for embeddings and AI-generated answers
- 📦 **Pinecone** for vector similarity search
- ⚙️ **Flask** for the web backend

---

## 🚀 Features

- Upload one or more PDF files
- Extracts and chunks the content
- Converts chunks to semantic embeddings using Gemini
- Stores embeddings in Pinecone under a per-user namespace
- Lets users ask questions grounded in their uploaded documents
- Retrieves relevant chunks and generates answers with Gemini chat model

---

## 🧰 Tech Stack

| Component     | Tool |
|---------------|------|
| Embedding     | Google Gemini (`text-embedding-004`) |
| Vector DB     | Pinecone |
| LLM           | Google Gemini (`gemini-2.0-flash`) |
| Backend       | Python + Flask |
| PDF Parsing   | pdfplumber |
| Orchestration | LangChain (chunking, schema) |
| Env Mgmt      | python-dotenv |

---

## 🛠️ Setup Instructions

### 1. Clone the repository
'''bash
git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf

### 2. Create and activate virtual environment
- python -m venv env
- source env/bin/activate   # On Windows: env\Scripts\activate

### 3.Create .env file
- GOOGLE_API_KEY=your_google_api_key
- PINECONE_API_KEY=your_pinecone_api_key


