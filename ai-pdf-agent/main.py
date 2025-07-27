import os
from dotenv import load_dotenv
import google.generativeai as genai

from backend.loader import extract_text_from_pdfs
from backend.chunking import chunk_text
from backend.embedding import create_vector_store, load_vector_store
from backend.qa_handler import get_chain, get_fallback_answer

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def run_conversational_pdf_chat(pdf_paths):
    # Load and embed PDF content
    pdf_files = [open(path, "rb") for path in pdf_paths]
    raw_text = extract_text_from_pdfs(pdf_files)
    chunks = chunk_text(raw_text)
    create_vector_store(chunks)
    vector_store = load_vector_store()

    # Create conversational chain with memory
    chain = get_chain()

    print("\n✅ PDF processed. You can now start chatting with it!")
    print("💬 Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        user_question = input("🧠 You: ").strip()
        if user_question.lower() in ("exit", "quit"):
            print("👋 Exiting the chat. Goodbye!")
            break

        docs = vector_store.similarity_search(user_question, k=3)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response["output_text"]

        if "Answer not found in the provided context." in answer:
            print("🔍 Info not found in documents. Asking Gemini directly...")
            answer = get_fallback_answer(user_question)

        print(f"🤖 Doc Agent : {answer}\n")

if __name__ == "__main__":
    run_conversational_pdf_chat(["example_two.pdf"])
