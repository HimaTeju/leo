import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize LangChain and model
st.set_page_config(page_title="FinVA - Financial Virtual Assistant", layout="wide")

template = """
Answer the question below.

Here is the information about you: {context}

Question: {question}

Answer:
"""
context = """
Your name is FinVA - Financial Virtual Assistant.
You provide real-time financial data, stock market trends, and investing tips.
"""
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.2")
chain = prompt | model

def get_financial_response(question):
    return chain.invoke({"context": context, "question": question})

# Streamlit UI
st.title("💰 FinVA - Financial Virtual Assistant")
st.write("Your AI-powered assistant for financial data and insights.")

# Chat interface with history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.header("Chat with FinVA")
user_query = st.text_input("Ask me anything about finance or investing:")
if st.button("Send") and user_query:
    with st.spinner("Thinking..."):
        response = get_financial_response(user_query)
        st.session_state.chat_history.append((user_query, response))

# Display chat history
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"<div style='text-align: right;'><b>You:</b> {user_msg}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: left; background-color: #f1f1f1; padding: 10px; border-radius: 5px;'><b>FinVA:</b> {bot_msg}</div>", unsafe_allow_html=True)
