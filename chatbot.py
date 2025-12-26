import streamlit as st
import requests
import os

# -------------------------
# CONFIG
# -------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = "phi3"

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="MuseApp Chatbot")

st.title("🎨 MuseApp Chatbot")
st.write("Cloud UI + Local LLaMA (via Cloudflare)")

# -------------------------
# Conversation memory
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type a message")

if user_input:
    st.session_state.chat.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    payload = {
        "model": MODEL,
        "messages": st.session_state.chat,
        "stream": False
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=120
    )

    answer = response.json()["message"]["content"]

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat.append(
        {"role": "assistant", "content": answer}
    )
