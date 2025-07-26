from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

QA_PROMPT_TEMPLATE = """
Answer the question using the context below. Be detailed and accurate. If answer is not in the context, say:
"Answer not found in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

def get_chain():
    api_key = os.getenv("GOOGLE_API_KEY")
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_answer(vector_store, question, k=3):
    docs = vector_store.similarity_search(question, k=k)
    chain = get_chain()
    result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return result["output_text"]
