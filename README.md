# 🎓 AI-Based Chatbot for Student Services

> **IBM Project-Based Experiential Learning (PBEL) 2026**
> An AI-powered Retrieval-Augmented Generation (RAG) chatbot that helps students quickly access information from official university documents using semantic search and a Large Language Model.

---

## 📖 Overview

Finding information in university manuals, academic regulations, and student handbooks can be time-consuming. This project provides an intelligent chatbot that understands natural language questions, retrieves the most relevant sections from official university documents, and generates accurate, context-aware responses.

Unlike a traditional chatbot that relies only on pre-trained knowledge, this application uses **Retrieval-Augmented Generation (RAG)** to ensure that every response is grounded in the uploaded university documents.

---

# ✨ Features

* 📄 Supports PDF and TXT documents
* 🤖 Retrieval-Augmented Generation (RAG)
* 🧠 Semantic search using Sentence Transformers
* 🔍 Context-aware document retrieval
* 💬 AI-generated responses using **Qwen**
* 📚 Displays retrieved source passages with similarity scores
* ⚡ Fallback to retrieved context when generation is unavailable
* 🎯 Responses grounded only in uploaded documents
* 🖥️ Simple and interactive Streamlit interface

---

# 🏗️ System Architecture

```text
                 Student Question
                        │
                        ▼
              Sentence Embedding Model
                        │
                        ▼
             Semantic Vector Search
                        │
                        ▼
          Most Relevant Document Chunks
                        │
                        ▼
      Prompt Construction with Retrieved Context
                        │
                        ▼
                 Qwen Language Model
                        │
                        ▼
            AI Generated Grounded Answer
                        │
                        ▼
      Retrieved Evidence + Similarity Scores
```

---

# 🛠️ Tech Stack

| Category             | Technology                           |
| -------------------- | ------------------------------------ |
| Language             | Python                               |
| Frontend             | Streamlit                            |
| Embedding Model      | all-MiniLM-L6-v2                     |
| Large Language Model | Qwen/Qwen2.5-1.5B-Instruct           |
| Similarity Search    | Cosine Similarity                    |
| Document Processing  | PyPDF2                               |
| AI Technique         | Retrieval-Augmented Generation (RAG) |

---

# 📂 Project Structure

```text
AI-Based-Chatbot-for-Student-Services/
│
├── app.py
├── rag_engine.py
├── requirements.txt
├── README.md
│
└── sample_documents/
    ├── Student Manual.pdf
    ├── AICTE Curriculum.pdf
    └── University Handbook.txt
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/ES1307/AI-Based-Chatbot-for-Student-Services.git
```

Move into the project

```bash
cd AI-Based-Chatbot-for-Student-Services
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 💡 Example Questions

* What is the attendance requirement?
* What is the examination policy?
* How can I apply for scholarships?
* What are the library timings?
* What student support services are available?
* What documents are required during admission?

---

# 🔄 RAG Workflow

1. Upload university documents.
2. Extract text from PDFs and TXT files.
3. Split documents into overlapping chunks.
4. Generate vector embeddings.
5. Convert the user's question into an embedding.
6. Retrieve the most relevant document chunks.
7. Build a grounded prompt using the retrieved context.
8. Generate the final response with **Qwen**.
9. Display retrieved evidence and similarity scores.

---

# 📸 Screenshots

Add screenshots of:

* Home Screen
* Chat Interface
* Retrieved Context
* Generated Answer
* Source Evidence

---

# 🎯 Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Embeddings
* Prompt Engineering
* Large Language Models (LLMs)
* Explainable AI
* Information Retrieval
* Responsible AI Practices

---

# 🚀 Future Improvements

* FAISS or ChromaDB vector database
* Multi-turn conversation memory
* OCR support for scanned PDFs
* Voice-based interaction
* Multi-language support
* REST API
* User authentication
* Cloud deployment (Render/Hugging Face Spaces/Azure)

---

# 📄 Dataset

The chatbot retrieves information from official university documents included in the project, such as:

* Student Handbook
* AICTE Curriculum
* University Academic Documents

These documents are indexed at runtime to generate grounded responses.

---

# ⚠️ Disclaimer

Responses are generated only from the uploaded documents. Users should always verify important academic information with the latest official university notifications.

---

# 👨‍💻 Author

**Eshaan Sabharwal**

B.Tech – Computer Science & Engineering

Shri Ram Murti Smarak College of Engineering & Technology

IBM Project-Based Experiential Learning (PBEL)

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

