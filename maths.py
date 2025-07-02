# 1. Required Libraries import karo
import streamlit as st
import json, os
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Groq se LLM model load krne k liye
from langchain.memory import ConversationBufferMemory  # Memory for chat history
from langchain.chains import ConversationChain  # Chat chain create krne k liye

# 2. Load .env file (agr env use kar rahe ho)
load_dotenv()

# 3. Streamlit secrets se API Key lo (secure tarika)
groq_api_key = st.secrets["GROQ_API_KEY"]  # yahan se .streamlit/secrets.toml file se key uthay ga

# 4. Streamlit App Setup
st.set_page_config(page_title="Math's Tutor ChatBOT")  # browser tab title
st.title("MATH'S TUTOR BOT")  # app ka main heading

# 5. Sidebar Controls (model selection and token limit)
model_name = st.sidebar.selectbox(
    "🧠 Select Groq Model",
    ["gemma2-9b-it", "deepseek-r1-distill-llama-70b", "llama3-8b-8192"]
)

max_tokens = st.sidebar.slider(
    "🔢 Max Tokens", 50, 300, 150  # min, max, default
)

# 6. Chat Memory initialize kro (sirf ek bar)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)  # chat messages yaad rakhega

# 7. History initialize kro (display k liye)
if "history" not in st.session_state:
    st.session_state.history = []  # empty list to store all messages in order

# 8. 📂 Button: Load Previous Chat
if st.sidebar.button("📂 Load Previous Chat"):
    if os.path.exists("chat_history.json"):  # file agar mojood ho
        with open("chat_history.json", "r") as file:
            st.session_state.history = json.load(file)  # history load krlo
        st.success("✅ Previous chat loaded!")
    else:
        st.warning("⚠️ No previous chat found.")

# 9. 🗃️ Function: Save current chat to file
def save_chat():
    with open("chat_history.json", "w") as file:
        json.dump(st.session_state.history, file)  # current history save krdo

# 10. 💬 User input from chat bar
user_input = st.chat_input("You: ")  # chat box show hoga niche

# 11. 👂 Agar user ne kuch likha
if user_input and isinstance(user_input, str):

    # (1) user ka message history main save kro
    st.session_state.history.append(("user", user_input))

    # (2) model initialize kro selected model k sath
    llm = ChatGroq(
        model_name=model_name,
        max_tokens=max_tokens,
        groq_api_key=groq_api_key
    )

    # (3) ConversationChain build kro with memory
    conv = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False  # backend logs hide rakhni ho to False
    )

    # (4) AI se response lo
    try:
        ai_response = conv.predict(input=user_input)
    except Exception as e:
        ai_response = f"⚠️ Error: {str(e)}"

    # (5) AI ka jawab bhi history mein add kro
    st.session_state.history.append(("assistant", ai_response))

    # (6) Chat ko file mein save kro
    save_chat()

# 12. 🗨️ Show Chat Messages in Bubble Format
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text)  # user bubble
    else:
        st.chat_message("assistant").write(text)  # assistant bubble


