import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import torch

# LangChain modules
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai

# ---------------------- LOAD ENVIRONMENT ----------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# ---------------------- PDF HANDLING ----------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    # Use Hugging Face for embeddings (local)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# ---------------------- RETRIEVAL CHAIN ----------------------
def get_retrieval_chain():
    # Load FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Prompt for Gemini
    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent assistant. Use the provided context to answer the question clearly and concisely.
    If the answer is not available in the context, respond: "The answer is not available in the provided context."

    Context:
    {context}

    Question:
    {question}
    """)

    # Gemini model for answering
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

    # Build chain
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return retrieval_chain


# ---------------------- USER INPUT ----------------------
def user_input(question):
    chain = get_retrieval_chain()
    response = chain.invoke(question)
    st.write("**Answer:**", response)


# ---------------------- MAIN APP ----------------------
def main():
    st.set_page_config(page_title="Chat with PDF (Hybrid)", layout="centered")
    st.title("📘 Chat with Multiple PDFs (Hugging Face + Gemini)")

    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(text)
                    get_vector_store(chunks)
                    st.success("PDFs processed successfully ✅")
            else:
                st.warning("Please upload at least one PDF.")

    user_question = st.text_input("Ask a question about your PDFs:")
    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()
