from PyPDF2 import PdfReader

def extract_text_from_pdfs(pdf_files):
    """
    Accepts a list of file-like PDF objects and extracts page-level text with metadata.
    Returns a list of dicts: {"text": ..., "metadata": {"source": ..., "page": ...}}
    """
    documents = []

    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        file_name = getattr(pdf, "name", "Unknown")
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            documents.append({
                "text": text,
                "metadata": {
                    "source": file_name,
                    "page": i + 1
                }
            })

    return documents
