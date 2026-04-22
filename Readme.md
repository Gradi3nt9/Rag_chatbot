# PDF RAG Chatbot

A simple Streamlit app that lets you upload PDF files and chat with them using a Retrieval-Augmented Generation (RAG) pipeline built with LangChain, Chroma, and Groq.

## Features
- Upload one or more PDF files
- Chunk and embed document content
- Store/reload embeddings with Chroma
- Ask questions with source page citations

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Groq API key:

```env
GROQ_API_KEY=your_api_key_here
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, upload PDFs, process them, and start chatting.
