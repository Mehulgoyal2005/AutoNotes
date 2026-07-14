# 🚀 AutoNotes AI 🎙️

AutoNotes AI is a full-stack AI-powered application that allows users to upload documents, interact with them using intelligent question-answering, and generate podcast-style audio summaries using AI.

It combines **Retrieval-Augmented Generation (RAG)** with **Groq's fast LLM inference API** to deliver quick, context-aware answers.

---

## 🌟 Project Overview

AutoNotes AI helps users:

* Understand large documents quickly
* Ask context-based questions
* Convert documents into engaging audio podcasts

This project demonstrates the integration of:

* AI (LLMs)
* NLP techniques
* Full-stack web development

---
## System Design
<img width="1536" height="1024" alt="System Design" src="https://github.com/user-attachments/assets/db612738-b0fc-4081-9b3a-3b3c1be3f488" />

## ✨ Key Features

### 💬 Document Chat (RAG System)

* 📄 Upload PDF documents
* 🤖 Ask questions based on document content
* 🔍 Semantic search using vector embeddings
* ⚡ Fast and accurate responses
* 🎯 Context-aware answers

---

### 🎙️ AI Podcast Generation

* 🎭 AI-generated conversations
* 🗣️ Natural speech using Text-to-Speech
* ⏱️ Short and engaging summaries
* 📥 Downloadable audio (MP3)

---

### 🎨 User Interface

* 🌓 Light/Dark mode
* ✨ Clean and modern UI
* 📱 Responsive design
* ⚡ Smooth user experience

---

## 🛠️ Tech Stack

### 🔹 Frontend

* React (TypeScript)
* CSS

### 🔹 Backend

* FastAPI (Python)
* FAISS (Vector Database)

### 🔹 AI / ML

* Groq API (LLM inference — Llama 3.3 70B)
* Sentence Transformers (embeddings)

### 🔹 Audio

* gTTS (Google Text-to-Speech)
* pydub + ffmpeg (audio processing)

---

## 🏗️ System Architecture

* PDF → Text Extraction
* Text → Chunking
* Chunking → Embeddings
* Stored in FAISS
* Query → Retrieve relevant chunks
* LLM generates final answer

---

## 🚀 Installation & Setup

### 🔹 Prerequisites

* [uv](https://docs.astral.sh/uv/) (Python package manager) — installs Python 3.10 automatically
* Node.js + npm
* ffmpeg (required for podcast audio): `brew install ffmpeg` (macOS)

### 🔹 1. Clone Repository

```bash
git clone https://github.com/Mehulgoyal2005/AutoNotes.git
cd AutoNotes
```

---

### 🔹 2. Setup Backend

From the project root:

```bash
uv sync
```

---

### 🔹 3. Configure API Key

Copy the example env file and add your [Groq API key](https://console.groq.com):

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY=your_key
```

The `.env` file must be at the **project root** (not inside `backend/`).

---

### 🔹 4. Start Backend

```bash
./start_backend.sh
```

Or manually:

```bash
source .venv/bin/activate
cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

### 🔹 5. Setup Frontend

```bash
cd frontend
npm install
npm start
```

---

## 🌐 Usage

1. Open → http://localhost:3000
2. Upload a PDF
3. Ask questions
4. Generate podcast

---

## 📁 Project Structure

```
AutoNotes/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   ├── models/
│   │   └── core/
│
├── frontend/
│   ├── src/
│   ├── public/
│
├── start_backend.sh
├── start_frontend.sh
└── README.md
```

---

## 🔮 Future Enhancements

* Multi-document support
* Chat history
* UI improvements
* Cloud deployment
* More AI model support

---

## 👩‍💻 Author

**Mehul Goyal**

---

## 🙏 Acknowledgements

* Inspiration from Google NotebookLM
* Groq for fast LLM inference
* FastAPI & React communities

---

## ⭐

If you like this project, give it a ⭐ on GitHub!
