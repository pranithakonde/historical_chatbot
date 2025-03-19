import os
import faiss
import pickle
import logging
import time
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)

VECTORSTORE_FILE = "../models/vectorstore.index"
EMBEDDINGS_FILE = "../models/embeddings.pkl"

if os.path.exists(VECTORSTORE_FILE):
    logging.info("Loading vector store...")
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
    vectorstore = FAISS.load_local(VECTORSTORE_FILE, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
else:
    logging.error("Vector store not found. Run process_pdf.py first.")
    exit()

llm = ChatGroq(
    groq_api_key="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",
    temperature=0.2,
    max_tokens=3000,
    model_kwargs={"top_p": 1}
)

template = """
<|context|>
You are a professional tour guide and historian with vast knowledge of historical places. Your task is to provide accurate and interesting details about various historical sites based on user queries.

Responsibilities:
- Respond to user queries about historical places with factual information.
- Include the significance, history, and any notable facts about the place.
- Suggest additional information for related historical landmarks if relevant.

Formatting Guidelines:
- Present the information in a structured format with bullet points where appropriate.
- Keep the responses clear and informative, maintaining a friendly and engaging tone.
- Ensure answers are concise, highlighting key facts and points of interest.

</s>
<|user|>
{query}
</s>
<|assistant|>
"""


prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_response(user_input, conversation_context=""):
    logging.info(f"Fetching response for: {user_input}")
    try:
        full_prompt = "Chat History: " + conversation_context + " User Input: " + user_input
        result = rag_chain.invoke(full_prompt)
        time.sleep(1)
        return result
    except Exception as e:
        logging.error(f"Error: {e}")
        return "Sorry, something went wrong."
