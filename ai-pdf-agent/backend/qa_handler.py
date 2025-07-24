from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

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
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
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
