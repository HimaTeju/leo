import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Set up the page
st.set_page_config(page_title="FinVA - Financial Virtual Assistant", layout="wide")

# LangChain initialization
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

# Add FAQ data
faq_data = [
    {"question": "Who am I?", "answer": "You are FinVA, your Financial Virtual Assistant."},
    {"question": "What is BSE & NSE?", "answer": "BSE (Bombay Stock Exchange) and NSE (National Stock Exchange) are the two primary stock exchanges in India. BSE is the oldest stock exchange in Asia, while NSE is known for its advanced electronic trading platform."},
    {"question": "What is a stock market index?", "answer": "A stock market index is a measurement of a section of the stock market. It is computed from the prices of selected stocks and is used as a benchmark to compare the performance of individual stocks or portfolios."}
]

# Streamlit UI
st.title("💰 FinVA - Financial Virtual Assistant")
st.write("Your AI-powered assistant for financial data and insights.")

# Chat interface with history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.header("Chat with FinVA")
st.write("Ask me anything about finance or investing!")

# FAQ Section within Chat
st.subheader("Common Questions You Can Ask:")
faq_columns = st.columns(len(faq_data))
for i, faq in enumerate(faq_data):
    with faq_columns[i]:
        if st.button(faq["question"], key=f"faq_{i}"):
            response = get_financial_response(faq["question"])
            st.session_state.chat_history.append((faq["question"], response))

# User Input
user_query = st.text_input("Type your question:", placeholder="e.g., What are today's stock trends?")
if st.button("Send", key="user_query") and user_query:
    with st.spinner("Thinking..."):
        response = get_financial_response(user_query)
        st.session_state.chat_history.append((user_query, response))

# Display chat history
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"<div style='text-align: right;'><b>You:</b> {user_msg}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: left; background-color: #f9f9f9; padding: 10px; border-radius: 5px;'><b>FinVA:</b> {bot_msg}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 12px;'>💡 **Tip:** Swipe through the common questions or ask your own!</div>",
    unsafe_allow_html=True
)
