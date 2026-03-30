import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os

# -------------------------------
# 1️⃣ Set up your environment key
# -------------------------------
# Ensure you have set this in your terminal or .env file
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="centered")

if not groq_api_key:
    st.error("🚨 GROQ_API_KEY not found. Please set it in your environment variables.")
    st.stop()

# -------------------------------
# 2️⃣ Initialize the Groq model
# -------------------------------
try:
    # Use a verified model ID. 
    # Options: "llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=groq_api_key,
        streaming=True
    )
except Exception as e:
    st.error(f"Failed to initialize Groq model: {e}")
    st.stop()

# -------------------------------
# 3️⃣ Chat History Management
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]

st.title("🤖 Groq LLM Chatbot")

# Sidebar for controls
with st.sidebar:
    if st.button("Clear Conversation"):
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
        st.rerun()

# -------------------------------
# 4️⃣ Display Chat History
# -------------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    # We skip rendering the SystemMessage to keep the UI clean

# -------------------------------
# 5️⃣ User Input & Streaming
# -------------------------------
user_query = st.chat_input("Type your message...")

if user_query:
    # Add user message to history and show it
    st.session_state.messages.append(HumanMessage(content=user_query))
    st.chat_message("user").write(user_query)

    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Stream the response
        try:
            for chunk in llm.stream(st.session_state.messages):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            # Final update without the cursor
            response_placeholder.markdown(full_response)
            
            # Crucial: Save as AIMessage, not SystemMessage
            st.session_state.messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
