# 🚀 AutoNotes AI 🎙️

AutoNotes AI is a full-stack AI-powered application that allows users to upload documents, interact with them using intelligent question-answering, and generate podcast-style audio summaries using AI.

It combines **Retrieval-Augmented Generation (RAG)** with **local LLMs (Ollama)** to provide a completely offline-capable AI experience.

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
## Sytem Design 
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
* LangChain

### 🔹 AI / ML

* Ollama (Local LLM runtime)
* LLaMA 3 model
* Sentence Transformers

### 🔹 Audio

* gTTS / ElevenLabs (optional)

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

### 🔹 1. Clone Repository

```bash
git clone https://github.com/shravaniranee/AutoNotes.git
cd AutoNotes
```

---

### 🔹 2. Setup Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 🔹 3. Setup Ollama

Install Ollama and run:

```bash
ollama run llama3
```

---

### 🔹 4. Start Backend

```bash
uvicorn app.main:app --reload
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

**Shravani Pradeep Rane**

---

## 🙏 Acknowledgements

* Inspiration from Google NotebookLM
* Ollama for local LLM support
* FastAPI & React communities

---

## ⭐

If you like this project, give it a ⭐ on GitHub!
