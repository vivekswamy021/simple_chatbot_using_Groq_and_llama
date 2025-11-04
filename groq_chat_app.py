import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os

# -------------------------------
# 1️⃣  Set up your environment key
# -------------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("🚨 GROQ_API_KEY environment variable not found. Please set it before running.")
    st.stop()

# -------------------------------
# 2️⃣  Initialize the Groq model
# -------------------------------
try:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key
    )
except Exception as e:
    st.error(f"Failed to initialize Groq model: {e}")
    st.stop()

# -------------------------------
# 3️⃣  Streamlit UI Layout
# -------------------------------
st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Groq LLM Chatbot")
st.markdown("Chat with **LLaMA 3.3 (70B)** model via **Groq** in real-time!")

# -------------------------------
# Clear Chat Button
# -------------------------------
if st.button("🧹 Clear Chat"):
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
    st.experimental_rerun()  # refresh the app to clear UI

# -------------------------------
# Store chat history
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]

# Display previous chat
for msg in st.session_state.messages[1:]:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, SystemMessage):
        st.chat_message("assistant").markdown(msg.content)

# Chat input
user_query = st.chat_input("Type your message...")

if user_query:
    st.session_state.messages.append(HumanMessage(content=user_query))
    st.chat_message("user").markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(st.session_state.messages):
            token = chunk.content
            full_response += token
            response_placeholder.markdown(full_response)

        st.session_state.messages.append(SystemMessage(content=full_response))
