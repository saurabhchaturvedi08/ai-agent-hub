from PyPDF2 import PdfReader

def extract_text_from_pdfs(pdf_files):
    """
    Accepts a list of file-like PDF objects and extracts their combined text.
    """
    full_text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = page.extract_text() or ""
            full_text += text
    return full_text
