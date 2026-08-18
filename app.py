"""
app.py
------
Streamlit front-end for the AI-Powered College Assistant (RAG-based).

Run with:
    streamlit run app.py
"""

import os
import streamlit as st

from rag_pipeline import (
    ingest_documents,
    retrieve_context,
    format_context_for_prompt,
    get_chroma_client,
    get_or_create_collection,
)
from llm_client import generate_answer

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI College Assistant",
    page_icon="🎓",
    layout="wide",
)

# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str}]

if "kb_ready" not in st.session_state:
    # Check if a collection already exists with data; otherwise flag for ingestion
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client, reset=False)
        st.session_state.kb_ready = collection.count() > 0
    except Exception:
        st.session_state.kb_ready = False


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 College Assistant")
    st.caption("AI-Powered RAG Assistant for Students")

    st.divider()

    st.subheader("📚 Knowledge Base")
    if st.session_state.kb_ready:
        st.success("Knowledge base is loaded and ready.")
    else:
        st.warning("Knowledge base not built yet.")

    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Processing college documents and building vector index..."):
            try:
                num_chunks = ingest_documents(reset=True)
                st.session_state.kb_ready = True
                st.success(f"Knowledge base rebuilt: {num_chunks} chunks indexed.")
            except Exception as e:
                st.error(f"Failed to build knowledge base: {e}")

    st.divider()

    st.subheader("⚙️ Settings")
    top_k = st.slider("Number of context chunks to retrieve", 2, 8, 4)
    show_sources = st.checkbox("Show retrieved sources", value=True)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Categories covered: Department Info • Syllabus • "
        "Exam Guidelines • Academic Calendar • Student Activities"
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY is not set. Add it as an environment variable.")


# --------------------------------------------------------------------------
# Main chat interface
# --------------------------------------------------------------------------
st.title("🎓 AI-Powered College Assistant")
st.caption(
    "Ask me anything about departments, syllabus, exams, the academic "
    "calendar, or student activities — I'll answer using the official "
    "college knowledge base."
)

# Example quick-start prompts
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    example_cols = st.columns(3)
    examples = [
        "What is the attendance requirement for exams?",
        "Tell me about the CSE department.",
        "When does the semester break start?",
    ]
    for col, example in zip(example_cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state.pending_query = example

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources") and show_sources:
            with st.expander("📖 Sources used"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['source']}** (relevance: {src['score']:.2f})")

# Handle input (either typed or from example button)
user_input = st.chat_input("Ask a question about your college...")
if "pending_query" in st.session_state:
    user_input = st.session_state.pop("pending_query")

if user_input:
    if not st.session_state.kb_ready:
        st.error("Please build the knowledge base first using the sidebar button.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching the knowledge base and generating an answer..."):
                try:
                    retrieved = retrieve_context(user_input, top_k=top_k)
                    context_str = format_context_for_prompt(retrieved)

                    # Build short chat history for conversational context
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[-6:-1]
                    ]

                    answer = generate_answer(
                        query=user_input,
                        context=context_str,
                        chat_history=history,
                    )
                except Exception as e:
                    answer = f"Sorry, something went wrong while generating a response: {e}"
                    retrieved = []

                st.markdown(answer)
                if retrieved and show_sources:
                    with st.expander("📖 Sources used"):
                        for src in retrieved:
                            st.markdown(f"- **{src['source']}** (relevance: {src['score']:.2f})")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved,
        })