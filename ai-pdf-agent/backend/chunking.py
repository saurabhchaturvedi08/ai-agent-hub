from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for chunk in splits:
            all_chunks.append({
                "text": chunk,
                "metadata": doc["metadata"]
            })

    return all_chunks
