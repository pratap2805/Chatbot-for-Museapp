import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import ollama

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="MuseApp Chatbot", layout="centered")
st.title("🎭 MuseApp Assistant")
st.caption("Local conversational AI for Artists & Customers")

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
        st.markdown(msg["content"])

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
        st.markdown(user_input)

    # Retrieve KB context
    context = retrieve_context(user_input)

    # Build conversational history (last 6 messages)
    history = ""
    for msg in st.session_state.messages[-6:]:
        history += f"{msg['role'].upper()}: {msg['content']}\n"

    # -------------------- PROMPT (phi3 friendly) --------------------
    prompt = f"""
You are MuseApp's AI assistant.

User role: {role}

Be friendly, conversational, and concise.
Remember previous messages.
If unsure, ask a clarifying question.

Knowledge Base:
{context}

Conversation so far:
{history}

Now answer the user's latest message.
"""

    # -------------------- OLLAMA CALL (phi3) --------------------
    response = ollama.chat(
        model="phi3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"]

    # Store assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

    # -------------------- OPTIONAL BOOKING LINK --------------------
    if "book" in user_input.lower():
        st.markdown("🔗 **Book here:** https://museapp.com/book/artist123")
