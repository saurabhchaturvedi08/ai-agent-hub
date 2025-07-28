from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 🔄 Smarter prompt that lets the model decide to add its own knowledge if needed
QA_PROMPT_TEMPLATE = """
You are a helpful AI assistant.

Answer the question using the context provided below. If the context contains enough information, rely only on it.
If the context is not sufficient or relevant, enhance your answer using your own general knowledge to be as helpful and accurate as possible.

Context:
{context}

Question:
{question}

Answer:
"""

# Shared conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question"
)

# QA chain with fallback logic handled in prompt
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

# Optional: direct fallback model call if needed elsewhere
def get_fallback_answer(question):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    return model.invoke(question)
