import streamlit as st
import requests
import os

# =========================
# CONFIG
# =========================
OLLAMA_URL = os.getenv("OLLAMA_URL")  # from Streamlit Secrets
st.write("Using Ollama URL:", OLLAMA_URL)

MODEL = "phi3"

st.set_page_config(
    page_title="MuseApp Chatbot",
    page_icon="🎨"
)

# =========================
# SAFETY CHECK
# =========================
if not OLLAMA_URL:
    st.error("❌ OLLAMA_URL is not set. Please configure it in Streamlit Secrets.")
    st.stop()

# =========================
# UI HEADER
# =========================
st.title("🎨 MuseApp Chatbot")
st.caption("Cloud UI • Local LLaMA (via HTTP)")

# =========================
# CONVERSATION MEMORY
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = [
        {
            "role": "assistant",
            "content": "Hi! I’m the MuseApp assistant. How can I help you today?"
        }
    ]

# Display chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================
# USER INPUT
# =========================
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.chat.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    # =========================
    # CALL OLLAMA (HTTP)
    # =========================
    payload = {
        "model": MODEL,
        "messages": st.session_state.chat,
        "stream": False
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=120
        )
    except requests.exceptions.RequestException as e:
        st.error("❌ Failed to connect to Ollama")
        st.text(str(e))
        st.stop()

    # =========================
    # HANDLE NON-200 RESPONSES
    # =========================
    if response.status_code != 200:
        st.error(f"❌ Ollama returned HTTP {response.status_code}")
        st.text(response.text)
        st.stop()

    # =========================
    # SAFE JSON PARSING
    # =========================
    try:
        data = response.json()
        answer = data["message"]["content"]
    except Exception:
        st.error("❌ Ollama response was not valid JSON")
        st.text(response.text)
        st.stop()

    # =========================
    # DISPLAY ASSISTANT MESSAGE
    # =========================
    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat.append(
        {"role": "assistant", "content": answer}
    )
