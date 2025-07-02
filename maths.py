# 1. Import required libraries
import streamlit as st
from dotenv import load_dotenv  # .env file read krne k liye
import os
import json
from datetime import datetime
from langchain_groq import ChatGroq  #Groq API se LLM model load krne k liye
from langchain.memory import ConversationBufferMemory  #Memory backend for chat
from langchain.chains import ConversationChain  ##chain that wires LLM + memory

# 2. Load API KEY from .env file
load_dotenv()  #read .env file
groq_api_key = st.secrets["GROQ_API_KEY"]  #securely get key from secrets

# 3. Streamlit App setup
st.set_page_config(page_title="Math's Tutor ChatBOT")  #tab ka title
st.title("MATH'S TUTOR BOT📚")

# 4. Create chat folder if not exist
if not os.path.exists("chats"):
    os.makedirs("chats")

# 5. Sidebar Controls
model_name = st.sidebar.selectbox(
    "Select Groq Model",
    ["gemma2-9b-it", "deepseek-r1-distill-llama-70b", "llama3-8b-8192"]
)
max_tokens = st.sidebar.slider(
    "Max Tokens", 50, 300, 150  #min,max,default
)

# 6. Sidebar: Load previous chats
chat_files = sorted([f for f in os.listdir("chats") if f.endswith(".json")])
selected_chat = st.sidebar.selectbox(" Load Previous Chat", [""] + chat_files)

if st.sidebar.button("📤 Load Selected Chat") and selected_chat:
    with open(f"chats/{selected_chat}", "r") as f:
        st.session_state.history = json.load(f)
    st.success(f"Loaded: {selected_chat}")

# 7. Sidebar: New chat button
if st.sidebar.button("🆕 New Chat"):
    st.session_state.history = []
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.success("✨ New chat started!")

# 8. Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# 9. Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# 10. User input
user_input = st.chat_input("You : ")

if user_input and isinstance(user_input, str):
    st.session_state.history.append(("user", user_input))  #store user msg

    # 11. LLM initialization
    llm = ChatGroq(
        model_name=model_name,
        max_tokens=max_tokens
    )

    # 12. Build conversation chain
    conv = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )

    # 13. Get AI response
    try:
        ai_response = conv.predict(input=user_input)
    except Exception as e:
        ai_response = f"⚠️ Error: {str(e)}"

    # 14. Add AI response to history
    st.session_state.history.append(("assistant", ai_response))

    # 15. Auto-save current chat in unique file
    chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
    with open(f"chats/chat_{chat_id}.json", "w") as f:
        json.dump(st.session_state.history, f)

# 16. Display chat bubble
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)




