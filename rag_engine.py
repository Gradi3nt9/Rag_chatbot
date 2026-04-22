from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.prompts import PromptTemplate
from pathlib import Path
from dotenv import load_dotenv
import os


def load_pdfs(pdfs: str):
    docs = []
    for filename in os.listdir(pdfs):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(pdfs, filename))
            docs.extend(loader.load())
    return docs

def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""])
    
    chunks = splitter.split_documents(documents)
    return chunks

def build_vector_store(chunks, persist_dir: str = "./chroma_db"):
    # Create embeddings using HuggingFace's model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    # Create a Chroma vector store from the chunks and their embeddings
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=persist_dir)
    
    return vectorstore

def load_existing_vector_store(persist_dir: str = "./chroma_db"):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    return Chroma(
        embedding_function=embedding_model, 
        persist_directory=persist_dir
    )

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

def build_rag_chain(vectorstore):
    # Initialize the Groq LLM with your API key from .env
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model = "llama-3.3-70b-versatile",
        temperature=0.2, # lower = more factual, less creative. Good for RAG.
    )

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",    
        return_messages=True,
        output_key="answer"
    )

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(

        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 20} 
    )

    qa_prompt = PromptTemplate.from_template(
        """You are a helpful assistant that answers questions based on the provided documents.

Conversation history:
{chat_history}

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based only on the context provided above.
- If the context doesn't contain enough information to answer, say so clearly. Do not make things up.
- At the end of your answer, list the sources you used in this format: [Source: filename, Page X]
- Be concise but complete.

Answer:"""
    )

    chain= ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True, # gives us the actual chunks used, for citations
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    return chain


def query_rag(chain, question: str):
    result = chain.invoke({"question": question})
    
    answer = result["answer"]
    source_docs = result["source_documents"]
    
    # Extract unique sources for display
    sources = []
    seen = set()
    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append({"file": os.path.basename(source), "page": page + 1})
    
    return answer, sources
        
