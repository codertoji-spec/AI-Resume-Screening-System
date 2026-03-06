"""
resume_parser.py
----------------
Handles reading resume content from PDF files and plain text files.
Extracts raw text that will later be analyzed by the skill extractor
and embedding model.
"""

import io
import os

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


def extract_text_from_pdf(file_obj) -> str:
    """
    Extract all text from a PDF file object (BytesIO or file path).

    Parameters
    ----------
    file_obj : file-like object or str
        Either a BytesIO object (from Streamlit uploader) or a path string.

    Returns
    -------
    str
        All extracted text joined from every page.
    """
    if not PYPDF2_AVAILABLE:
        raise ImportError(
            "PyPDF2 is not installed. Run: pip install PyPDF2"
        )

    text_pages = []

    # Handle both file-path strings and binary file objects (BytesIO)
    if isinstance(file_obj, str):
        if not os.path.exists(file_obj):
            raise FileNotFoundError(f"PDF not found: {file_obj}")
        with open(file_obj, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
    else:
        # Assume file-like / BytesIO
        reader = PyPDF2.PdfReader(file_obj)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)

    return "\n".join(text_pages).strip()


def extract_text_from_txt(file_obj) -> str:
    """
    Read plain text from a .txt file object or path.

    Parameters
    ----------
    file_obj : file-like object or str

    Returns
    -------
    str
    """
    if isinstance(file_obj, str):
        with open(file_obj, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        raw = file_obj.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore").strip()
        return raw.strip()


def parse_resume(file_obj, filename: str) -> dict:
    """
    Unified entry point: detects file type and returns a dict with
    the candidate name and the full resume text.

    Parameters
    ----------
    file_obj : file-like object or str
    filename : str   – original file name, used to detect extension
                       and derive the candidate's display name.

    Returns
    -------
    dict with keys:
        - "name"  : str  candidate display name
        - "text"  : str  full resume text
        - "file"  : str  original filename
    """
    ext = os.path.splitext(filename)[-1].lower()
    candidate_name = os.path.splitext(os.path.basename(filename))[0].replace("_", " ").title()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_obj)
    elif ext in (".txt", ".md"):
        text = extract_text_from_txt(file_obj)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Please upload .pdf or .txt files."
        )

    return {
        "name": candidate_name,
        "text": text,
        "file": filename,
    }
