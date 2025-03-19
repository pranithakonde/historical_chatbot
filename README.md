Historical Chatbot

Overview

This project is a Retrieval-Augmented Generation (RAG) based Historical Chatbot that provides informative and engaging responses about historical places. It utilizes LangChain, FAISS, and HuggingFace Embeddings to process historical documents and answer user queries accurately.

Features

Processes and indexes historical documents using FAISS

Answers user queries with relevant historical details

Provides structured and engaging responses

Streamlit-based web interface for user interaction

Project Structure

Installation

1. Clone the repository

2. Create and activate a virtual environment

3. Install dependencies

Usage

1. Process Historical Documents

Before running the chatbot, process the historical PDF file to create embeddings:

2. Run the Chatbot

Start the chatbot web application using Streamlit:

How It Works

PDF Processing: process_pdf.py extracts text from the historical document, splits it into smaller chunks, and generates DistilBERT embeddings.

Vector Store: FAISS indexes the embeddings for efficient retrieval.

Query Processing: When a user enters a query, the chatbot retrieves relevant historical information using FAISS and generates a structured response.

Chat Interface: A simple Streamlit UI allows users to ask historical questions and receive detailed answers.

Configuration

Modify the following variables as needed:

PDF_FILE: Path to the historical document.

VECTORSTORE_FILE: Path to store the FAISS index.

EMBEDDINGS_FILE: Path to store the computed embeddings.

BATCH_SIZE: Number of text chunks processed in each batch.

Dependencies

Python 3.8+

FAISS

LangChain

HuggingFace Transformers

Streamlit

Pickle

Install them via:

Future Enhancements

Implement a chat history feature for better conversational flow.

Add support for multiple document sources.

Improve response formatting and citations.

Deploy as a web service using FastAPI or Flask.

License

This project is open-source under the MIT License.
