import os
import streamlit as st
from backend.loader import extract_text_from_pdfs
from backend.chunking import chunk_text
from backend.embedding import create_vector_store, load_vector_store
from backend.qa_handler import get_chain, get_fallback_answer
from langchain.schema import HumanMessage, AIMessage
import time

UPLOAD_DIR = "uploads"
VECTOR_DIR = "vector_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

st.set_page_config(page_title="📄 DocuMind - Chat with Your PDFs", layout="wide")
st.title("📄 DocuMind")
st.caption("Chat with multiple PDFs using Google Gemini & LangChain")

# Session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chain" not in st.session_state:
    st.session_state.chain = get_chain()
if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []

# Upload PDFs
uploaded_files = st.file_uploader("📁 Upload PDF(s)", type="pdf", accept_multiple_files=True)

# Mode selector
fallback_mode = st.radio("🤖 Answer Mode", [
    "Only answer from uploaded documents",
    "Also use Gemini if answer not found in documents"
])

# PDF Changed? Check hash via names and sizes
def files_changed(files):
    prev = [(f.name, f.size) for f in st.session_state.last_uploaded_files]
    curr = [(f.name, f.size) for f in files]
    return prev != curr

# Process PDFs (only if changed)
if uploaded_files and files_changed(uploaded_files):
    st.session_state.last_uploaded_files = uploaded_files
    with st.spinner("🔍 Reading & embedding PDFs..."):
        file_paths = []
        for file in uploaded_files:
            path = os.path.join(UPLOAD_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.read())
            file_paths.append(path)

        pdf_file_objs = [open(p, "rb") for p in file_paths]
        try:
            raw_text = extract_text_from_pdfs(pdf_file_objs)
            chunks = chunk_text(raw_text)
            create_vector_store(chunks)  # saves to disk
            st.session_state.vector_store = load_vector_store()
            st.success("✅ Documents processed and ready!")
        finally:
            for f in pdf_file_objs:
                f.close()
else:
    if uploaded_files and st.session_state.vector_store is None:
        st.session_state.vector_store = load_vector_store()

# Chat input box
question = st.chat_input("Ask something about your documents...")
if question and st.session_state.vector_store:
    with st.chat_message("user"):
        st.markdown(question)

    # Save user query to history
    st.session_state.chat_history.append(HumanMessage(content=question))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = st.session_state.vector_store.similarity_search(question, k=3)
            chain = st.session_state.chain

            response = chain({
                "input_documents": docs,
                "question": question,
                "chat_history": st.session_state.chat_history  # pass memory
            }, return_only_outputs=True)

            answer = response["output_text"]

            # Fallback if needed
            if "Answer not found in the provided context." in answer and fallback_mode == "Also use Gemini if answer not found in documents":
                answer = get_fallback_answer(question)

            # Stream the response like ChatGPT
            placeholder = st.empty()
            full_response = ""
            for token in answer.split():
                full_response += token + " "
                placeholder.markdown(full_response + "▌")
                time.sleep(0.03)
            placeholder.markdown(full_response)

    st.session_state.chat_history.append(AIMessage(content=answer))

# Show full chat history (bottom-up style)
if st.session_state.chat_history:
    st.divider()
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)
