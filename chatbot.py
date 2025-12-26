import streamlit as st
import requests
import os

# =========================
# CONFIG
# =========================
MODEL = "phi3"
OLLAMA_URL = os.getenv("OLLAMA_URL")

st.set_page_config(
    page_title="MuseApp Chatbot",
    page_icon="🎨"
)

# =========================
# SAFETY CHECK
# =========================
if not OLLAMA_URL:
    st.error("❌ OLLAMA_URL is missing in Streamlit Secrets")
    st.stop()

# =========================
# HEADER
# =========================
st.title("🎨 MuseApp Chatbot")
st.caption("Cloud UI • Local LLaMA (via HTTP)")
st.write("Using Ollama URL:", OLLAMA_URL)

# =========================
# SESSION MEMORY
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = [
        {
            "role": "assistant",
            "content": "Hi! I’m the MuseApp assistant. How can I help you today?"
        }
    ]

# =========================
# DISPLAY CHAT HISTORY
# =========================
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================
# USER INPUT
# =========================
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state.chat.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    # =========================
    # BUILD PROMPT (MEMORY)
    # =========================
    prompt = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in st.session_state.chat
    )
    prompt += "\nassistant:"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    # =========================
    # OLLAMA CALL (WORKING)
    # =========================
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true"
            },
            timeout=120
        )
    except Exception as e:
        st.error("❌ Could not connect to Ollama")
        st.text(str(e))
        st.stop()

    # =========================
    # ERROR HANDLING
    # =========================
    if response.status_code != 200:
        st.error(f"❌ Ollama returned HTTP {response.status_code}")
        st.text(response.text)
        st.stop()

    try:
        data = response.json()
        answer = data.get("response", "").strip()
    except Exception:
        st.error("❌ Invalid JSON from Ollama")
        st.text(response.text)
        st.stop()

    # =========================
    # DISPLAY & SAVE RESPONSE
    # =========================
    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat.append(
        {"role": "assistant", "content": answer}
    )
