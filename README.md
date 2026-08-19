# 🎓 AI-Powered College Assistant (RAG-based)

An intelligent, conversational college assistant built with **Streamlit**,
**Groq LLM API**, and **ChromaDB** using a **Retrieval-Augmented Generation
(RAG)** pipeline. It answers student questions using a college-specific
knowledge base (departments, syllabus, exam guidelines, academic calendar,
student activities) instead of relying only on the LLM's general knowledge.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
Streamlit UI (app.py)
     │
     ▼
Retrieve relevant chunks  ──►  ChromaDB (vector store, local, persisted)
     │                              ▲
     │                              │
     │                    Embeddings (SentenceTransformers, local, free)
     │                              ▲
     │                              │
     │                    Chunked college documents (/data/*.txt)
     │
     ▼
Inject retrieved context into prompt
     │
     ▼
Groq LLM (llama-3.1-8b-instant)  ──►  Generates grounded answer
     │
     ▼
Answer + Sources shown in chat UI
```

## 📁 Project Structure

```
college_assistant/
├── app.py                 # Streamlit application (UI + chat logic)
├── rag_pipeline.py         # Document loading, chunking, embedding, retrieval
├── llm_client.py           # Groq API wrapper
├── requirements.txt        # Python dependencies
├── .env.example             # Template for API key
├── data/                   # College knowledge base (sample documents)
│   ├── department_info.txt
│   ├── syllabus.txt
│   ├── exam_guidelines.txt
│   ├── academic_calendar.txt
│   └── student_activities.txt
└── chroma_db/               # Auto-created: persisted vector database
```

## ⚙️ Setup

1. **Clone / download this project** and move into the folder:
   ```bash
   cd college_assistant
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get a free Groq API key:**
   - Sign up at https://console.groq.com/keys
   - Copy `.env.example` to `.env` and paste your key:
     ```
     GROQ_API_KEY=your_actual_key_here
     ```
   - Or set it directly in your terminal:
     ```bash
     export GROQ_API_KEY=your_actual_key_here      # Windows: set GROQ_API_KEY=...
     ```

5. **(Optional) Pre-build the vector database from the command line:**
   ```bash
   python rag_pipeline.py
   ```
   This is also available as a button inside the app itself.

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

On first run, click **"🔄 Rebuild Knowledge Base"** in the sidebar to
chunk, embed, and index the sample documents in `/data`. After that, just
ask questions in the chat box, e.g.:

- "What is the attendance requirement for exams?"
- "Tell me about the CSE department."
- "When does the odd semester end?"
- "What clubs can I join for coding?"

## 🧩 How the RAG Pipeline Works

1. **Load** — All `.txt` files in `/data` are read.
2. **Chunk** — Each document is split into ~700-character overlapping
   chunks so context isn't lost at boundaries (`rag_pipeline.chunk_text`).
3. **Embed** — Each chunk is converted into a vector using the local
   `all-MiniLM-L6-v2` SentenceTransformer model (no external API cost).
4. **Store** — Vectors + metadata (source file, chunk index) are stored
   in a persistent local ChromaDB collection.
5. **Retrieve** — On each user query, the top-k most similar chunks are
   fetched by cosine similarity.
6. **Generate** — The retrieved chunks are inserted into the LLM prompt
   as context, and Groq's LLM generates a grounded, natural-language
   answer, citing which source files were used.

## ➕ Adding Your Own College Data

Just drop more `.txt` files into the `/data` folder (or edit the existing
ones) and click **"Rebuild Knowledge Base"** in the sidebar — no code
changes needed. For PDFs/DOCX, you can extend `rag_pipeline.load_documents()`
to extract text before chunking (e.g. with `pypdf` or `python-docx`).

## 🚀 Suggested Next Features (Incremental Roadmap)

- 📄 Upload-your-own-document Q&A (per-student temporary knowledge base)
- 📝 Study-plan generator based on the syllabus + exam dates
- ❓ Important-question generator per subject
- 🗂️ Assignment/deadline tracker with reminders
- 🔐 Student login/authentication
- 🗣️ Voice input/output
- 🤖 Agent-based workflows (e.g., auto-check attendance eligibility)

## 🛠️ Tech Stack

| Component        | Technology                          |
|-------------------|--------------------------------------|
| UI / Frontend     | Streamlit                            |
| LLM               | Groq API (Llama 3.1 8B Instant)      |
| Embeddings        | SentenceTransformers (MiniLM-L6-v2)  |
| Vector Database   | ChromaDB (local, persistent)         |
| Language          | Python 3.9+                          |
