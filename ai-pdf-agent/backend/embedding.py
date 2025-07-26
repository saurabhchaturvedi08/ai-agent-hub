from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

EMBED_MODEL = "models/embedding-001"

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

def create_vector_store(chunks, persist_path="faiss_index"):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(persist_path)
    return persist_path

def load_vector_store(persist_path="faiss_index"):
    embeddings = get_embeddings()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)

