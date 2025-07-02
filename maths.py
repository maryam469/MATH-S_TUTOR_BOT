# 1. Import required libraries
import streamlit as st
from dotenv import load_dotenv  # .env file read krne k liye
import os
from langchain_groq import ChatGroq  #Groq API se LLM model load krne k liye
from langchain.memory import ConversationBufferMemory  #Memory backend for chat
from langchain.chains import ConversationChain  ##chain that wires LLM + memory

# 2. Load API KEY from .env file
load_dotenv()  #read .env file

groq_api_key = st.secrets["GROQ_API_KEY"]  #set groq key  #os.environ environment variable ko represent krti ha like secrets,credentials,password

# 3. Streamlit App setup
st.set_page_config(page_title="Math's Tutor ChatBOT")  #tab ka title
st.title("MATH'S TUTOR BOT")  #webpage ka heading

# 4. Sidebar Controls for model and token selection
model_name = st.sidebar.selectbox(
    "Select Groq Model",
    ["gemma2-9b-it", "deepseek-r1-distill-llama-70b", "llama3-8b-8192"]
)
max_tokens = st.sidebar.slider(
    "Max Tokens", 50, 300, 150  #min,max,default
)

# 5. Initialize memory to store conversation
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(   #previous messages store krne k liy
        return_messages=True
    )

# 6. Initialize chat history list
if "history" not in st.session_state:
    st.session_state.history = []

# 7. User input chat bar
user_input = st.chat_input("You : ")  #chat bar input

# 8. Agar user kuch likhta hai
if user_input and isinstance(user_input, str):
    st.session_state.history.append(("user", user_input))  #history main add krdo

    # 9. Initialize LLM Model from Groq
    llm = ChatGroq(
        model_name=model_name,
        max_tokens=max_tokens
    )

    # 10. Build ConversationChain with memory
    conv = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False  #True krne se backend logs show hoti hain
    )

    # 11. AI response predict kro
    try:
        ai_response = conv.predict(input=user_input)
    except Exception as e:
        ai_response = f"⚠️ Error: {str(e)}"  #error aae to show kro

    # 12. Assistant ka response bhi history main add kro
    st.session_state.history.append(("assistant", ai_response))

# 13. Show chat bubble history
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

