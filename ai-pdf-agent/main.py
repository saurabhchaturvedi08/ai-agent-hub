import os
from dotenv import load_dotenv

from backend.loader import extract_text_from_pdfs
from backend.chunking import chunk_text
from backend.embedding import create_vector_store, load_vector_store
from backend.qa_handler import get_answer

load_dotenv()

def run_pipeline(pdf_paths, question):
    # Load PDFs
    pdf_files = [open(path, "rb") for path in pdf_paths]
    raw_text = extract_text_from_pdfs(pdf_files)
    
    # Chunk it
    chunks = chunk_text(raw_text)
    
    # Store vectors
    create_vector_store(chunks)
    vector_store = load_vector_store()
    
    # Ask Question
    answer = get_answer(vector_store, question)
    print("\n🧠 Answer:\n", answer)

if __name__ == "__main__":
    run_pipeline(["example.pdf"], "What is the purpose of this document?")
