import streamlit as st
from chatbot import get_response

st.set_page_config(page_title="Historical Chatbot", layout="wide")

st.title("📜 Historical Chatbot")
st.write("Ask me anything about history!")

user_query = st.text_input("Enter your question:")
if user_query:
    response = get_response(user_query)
    st.write("**Response:**", response)
