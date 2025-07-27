from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

QA_PROMPT_TEMPLATE = """
Answer the question using the context below. Be detailed and accurate.
If the answer is not in the context, respond with:
"Answer not found in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

# Shared memory for the session
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question"
)

def get_chain():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt,
        memory=memory
    )

def get_fallback_answer(question):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    return model.invoke(question)
