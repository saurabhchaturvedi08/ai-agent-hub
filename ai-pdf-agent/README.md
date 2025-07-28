# 📄 DocuMind - Chat with Your PDFs

**DocuMind** is an AI-powered PDF assistant that lets you chat with one or more PDF documents using natural language. It extracts the content, embeds it using vector storage, and answers your queries using **Google Gemini** with intelligent fallback support when the answer isn't found in your documents.

---

## 🚀 Features

- 📁 Upload multiple PDF documents
- 🧠 Ask questions directly about the content
- 💬 Chat history maintained per session
- ⚡ Smart fallback to Gemini if documents lack the answer
- 📚 Document context-aware retrieval using LangChain + FAISS
- 🔁 Persistent vector reuse (no reprocessing unless files change)
- ✅ Streaming answers in top-to-bottom chat layout

---

## 🧰 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io)
- **LLM:** Google Gemini via `langchain-google-genai`
- **Embedding & Search:** FAISS Vector Store
- **PDF Handling:** PyPDF2
- **Backend Logic:** LangChain, Python

## 🧠 How It Works

1. User uploads one or more PDFs.
2. App extracts and chunks text from PDFs.
3. Chunks are embedded into a **FAISS** vector store.
4. When the user asks a question:
    - The app searches similar chunks using vector similarity.
    - It answers using **Google Gemini** with the retrieved context.
    - If no relevant answer is found and fallback is enabled, **Gemini** is used without document context.

---

## 🎛️ Answer Modes

### 🔹 Only answer from uploaded documents  
Gemini will **only use** the content from uploaded PDFs.

### 🔹 Also use Gemini if answer not found in documents  
If the documents **don’t contain the answer**, Gemini will **generate one directly** using its general knowledge.


## 🔮 Future Improvements

- ✅ **User authentication** & account-based vector stores  
- ✅ **Save and reload previous sessions**  
- 📈 **Export Q&A history** to CSV/JSON  
- 🌐 **Deployable** on Streamlit Cloud or Hugging Face Spaces  
- 🧩 **Plug-and-play support** for other LLMs like OpenAI, Claude, Mistral  

