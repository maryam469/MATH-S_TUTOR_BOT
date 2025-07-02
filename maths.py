# Required libraries import kro
import streamlit as st
import json, os
from dotenv import load_dotenv  # .env file se secret API keys load krne k liye
from langchain_groq import ChatGroq  # Groq API se LLM model load krne k liye
from langchain.memory import ConversationBufferMemory  # Chat ko yaad rakhne k liye memory
from langchain.chains import ConversationChain  # LLM + Memory ko connect krta hai

#.env file se environment variables load krne k liy
load_dotenv()

groq_api_key = st.secrets["GROQ_API_KEY"]


st.set_page_config(page_title="Math's Tutor ChatBOT")
st.title("MATH'S TUTOR BOT 🤖📚")

#Sidebar controls — Model name aur max token select kro
model_name = st.sidebar.selectbox(
    "🧠 Select Groq Model",  # dropdown ka title
    ["gemma2-9b-it", "deepseek-r1-distill-llama-70b", "llama3-8b-8192"]
)

max_tokens = st.sidebar.slider(
    "Max Tokens", 50, 300, 150  # (min, max, default)
)

#  Sidebar buttons for Load Chat, New Chat, and Save Chat
col1, col2, col3 = st.sidebar.columns(3)

#  Load previous chat from file
if col1.button("Load Chat"):
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r") as file:
            st.session_state.history = json.load(file)
        st.success("Previous chat loaded successfully!")
    else:
        st.warning("⚠️ No saved chat found.")

# Start new chat (reset memory and history)
if col2.button("New Chat"):
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.history = []
    if os.path.exists("chat_history.json"):
        os.remove("chat_history.json")
    st.success("✨ New chat started!")

# Save chat manually to file
if col3.button("Save Chat"):
    if "history" in st.session_state and st.session_state.history:
        with open("chat_history.json", "w") as file:
            json.dump(st.session_state.history, file)
        st.success(" Chat saved successfully!")
    else:
        st.warning("⚠️ No chat history to save.")


#  Memory initialize kro
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Chat history initialize 
if "history" not in st.session_state:
    st.session_state.history = []

# Function to save current chat to file
def save_chat():
    with open("chat_history.json", "w") as file:
        json.dump(st.session_state.history, file)

# User ka input (chat input box)
user_input = st.chat_input("You:")

# Agar user kuch likhe to process kro
if user_input and isinstance(user_input, str):
    # User ka message history mein save kr do
    st.session_state.history.append(("user", user_input))

    # LLM Model initialize kro (Groq se)
    llm = ChatGroq(
        model_name=model_name,
        max_tokens=max_tokens,
        groq_api_key=groq_api_key
    )

    # Conversation chain build kro (LLM + memory)
    conv = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False  # agar True karo to backend logs show honge
    )

    # LLM se AI response lo
    try:
        ai_response = conv.predict(input=user_input)
    except Exception as e:
        ai_response = f"⚠️ Error: {str(e)}"

    # Assistant ka message bhi history mein save kro
    st.session_state.history.append(("assistant", ai_response))

    # Puray chat ko file mein save kro
    save_chat()

#Chat history bubbles ko display kro
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)




