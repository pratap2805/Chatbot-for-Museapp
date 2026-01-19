import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import ollama

# -------------------- FORCE OLLAMA HOST --------------------
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

# -------------------- CONFIG --------------------
MODEL = "phi3"
MAX_HISTORY = 4  # keep small for speed

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="MuseApp Chatbot", layout="centered")
st.title("🎭 MuseApp Chatbot")
st.caption("Local AI Assistant for Artists & Customers")

# -------------------- ROLE SELECTION --------------------
role = st.radio("Who are you?", ["Customer", "Artist"], horizontal=True)

# -------------------- LOAD VECTOR STORE --------------------
@st.cache_resource
def load_vector_store():
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    return index, docs

index, documents = load_vector_store()

# -------------------- LOAD EMBEDDINGS --------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# -------------------- SESSION MEMORY --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- DISPLAY CHAT HISTORY --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------- VECTOR SEARCH --------------------
def retrieve_context(query, k=3):
    q_emb = embedder.encode([query])
    _, idx = index.search(q_emb, k)
    return "\n".join([documents[i] for i in idx[0]])

# -------------------- USER INPUT --------------------
user_input = st.chat_input("Ask something about MuseApp...")

if user_input:
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Retrieve KB context
    context = retrieve_context(user_input)

    # Short conversation history (FAST)
    history = "\n".join(
        m["content"]
        for m in st.session_state.messages[-MAX_HISTORY:]
    )

    # -------------------- CLEAN PROMPT --------------------
    prompt = f"""
You are the MuseApp assistant.

User type: {role}

Answer clearly and directly.
Do NOT repeat instructions or metadata.
Use the context only if helpful.

Context:
{context}

Conversation:
{history}

Respond to the user's last message.
"""

    # -------------------- OLLAMA CALL (LOCAL, STABLE) --------------------
    try:
        response = ollama.generate(
            model=MODEL,
            prompt=prompt,
            options={
                "temperature": 0.3,
                "num_predict": 200
            }
        )
        answer = response["response"].strip()

    except Exception as e:
        st.error("❌ Ollama is not reachable. Make sure `ollama run phi3` is running.")
        st.stop()

    # Store assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)

    # -------------------- OPTIONAL BOOKING CTA --------------------
    if "book" in user_input.lower():
        st.markdown("🔗 **Book an artist:** https://museapp.com/book")
