"""
llm_client.py
-------------
Thin wrapper around the Groq API for generating chat responses.
Keeping this in its own module makes it easy to swap providers later.
"""

import os
from groq import Groq

# Model choice: Llama 3.1 8B is fast and cheap; swap to a larger Groq-hosted
# model (e.g. llama-3.3-70b-versatile) if you need higher quality answers.
DEFAULT_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a helpful, friendly AI College Assistant.
You answer student questions using ONLY the context provided below,
which was retrieved from the college's official knowledge base
(department info, syllabus, exam guidelines, academic calendar, and
student activities).

Rules:
- Base your answer primarily on the provided context.
- If the context does not contain the answer, say clearly that the
  information is not available in the college knowledge base, and
  suggest the student contact the relevant department/office.
- Be concise, accurate, and speak directly to the student.
- Do not make up dates, names, or numbers that are not in the context.
"""


def get_client() -> Groq:
    """
    Create a Groq client. Expects the GROQ_API_KEY environment variable
    to be set (e.g. via `export GROQ_API_KEY=your_key_here` or a .env file).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable not set. "
            "Get a free key at https://console.groq.com/keys"
        )
    return Groq(api_key=api_key)


def generate_answer(query: str, context: str, chat_history=None,
                     model: str = DEFAULT_MODEL, temperature: float = 0.3) -> str:
    """
    Send the retrieved context + user query to the Groq LLM and return
    the generated answer text.

    chat_history: optional list of {"role": "user"/"assistant", "content": str}
                  for multi-turn conversational memory.
    """
    client = get_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history)

    user_turn = (
        f"Context from college knowledge base:\n{context}\n\n"
        f"Student question: {query}"
    )
    messages.append({"role": "user", "content": user_turn})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=800,
    )

    return response.choices[0].message.content
