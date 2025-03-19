import os
import faiss
import pickle
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)

PDF_FILE = "../data/historical_docs.pdf"
VECTORSTORE_FILE = "../models/vectorstore.index"
EMBEDDINGS_FILE = "../models/embeddings.pkl"
BATCH_SIZE = 32  # Adjust based on available memory


def process_pdf():
    logging.info("Loading PDF...")
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    logging.info("Initializing embedding model...")
    embeddings_model = HuggingFaceEmbeddings(model_name="distilbert-base-nli-stsb-mean-tokens")

    logging.info("Processing embeddings in batches...")
    batch_embeddings = []

    # Batch processing for embeddings
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_texts = [text.page_content for text in batch]  # Extract text content
        batch_embeds = embeddings_model.embed_documents(batch_texts)
        batch_embeddings.extend(batch_embeds)

    logging.info("Creating FAISS vector store...")
    # vectorstore = FAISS.from_embeddings(
    #     [(text.page_content, embedding) for text, embedding in zip(texts, batch_embeddings)]
    # )
    vectorstore = FAISS.from_embeddings(batch_embeddings, [text.page_content for text in texts])

    vectorstore.save_local(VECTORSTORE_FILE)

    logging.info("Saving embeddings...")
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(batch_embeddings, f)

    logging.info("Vector store and embeddings saved successfully.")


if __name__ == "__main__":
    process_pdf()
