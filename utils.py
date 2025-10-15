# utils.py
import os
import time
import logging
from typing import List

import PyPDF2
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ─────────────────────────────
# Environment setup
# ─────────────────────────────
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("utils")


def get_openai_client() -> OpenAI:
    """Create OpenAI client when needed (avoids slow import-time init)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not found in environment!")
    return OpenAI(api_key=api_key)


# ─────────────────────────────
# PDF text extraction
# ─────────────────────────────
def extract_text_from_pdf(file_path: str) -> List[tuple]:
    """
    Extract text from a PDF and return as list of (page_num, text).
    Also saves extracted text to 'extracted_text/<pdfname>.txt'.
    """
    pages_text = []
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                except Exception as e:
                    logger.warning(f"Page {page_num} extraction failed: {e}")
                    page_text = None
                if page_text:
                    pages_text.append((page_num, page_text.strip()))

        # Save for reference
        pdf_name = os.path.basename(file_path).replace(".pdf", "")
        os.makedirs("extracted_text", exist_ok=True)
        txt_path = os.path.join("extracted_text", f"{pdf_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for p_num, p_text in pages_text:
                f.write(f"[Page {p_num}]\n{p_text}\n\n")

        return pages_text

    except Exception as e:
        logger.error(f"Error extracting text from PDF '{file_path}': {e}")
        return []


# ─────────────────────────────
# Text chunking
# ─────────────────────────────
def chunk_text(text: str, max_tokens: int = 500) -> List[str]:
    """
    Splits text into smaller chunks based on token count.
    """
    if not text:
        return []
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
    return chunks


# ─────────────────────────────
# Fast Embedding Generator
# ─────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10),
       retry=retry_if_exception_type(Exception))
def get_openai_embedding(text: str, client: OpenAI = None) -> List[float]:
    """
    Generate OpenAI embeddings for given text.
    Uses 'text-embedding-3-small' for high speed and lower latency.
    Retries automatically on transient errors.
    """
    if client is None:
        client = get_openai_client()

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return []

    start = time.time()
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        duration = time.time() - start
        logger.debug(f"Embedding created in {duration:.2f}s (len={len(embedding)})")
        return embedding
    except Exception as e:
        logger.warning(f"Embedding API error: {e}")
        raise
