import hashlib
import os
import csv
from pathlib import Path
from typing import List, Dict, Any, Generator

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

import logging
logger = logging.getLogger(__name__)

# Very rough token approximation (1 token = ~4 chars)
CHUNK_SIZE_TOKENS = 512
OVERLAP_TOKENS = 100
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * 4
OVERLAP_CHARS = OVERLAP_TOKENS * 4


class DocumentParser:
    """Handles parsing and chunking of user documents for local RAG."""

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        """Splits text into overlapping chunks using character lengths as token approximation."""
        if not text:
            return []

        # Simple character-based striding
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + CHUNK_SIZE_CHARS, text_len)
            
            # Try to avoid splitting inside words if not at the very end
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                if last_space != -1 and last_space > start + CHUNK_SIZE_CHARS // 2:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            start = end - OVERLAP_CHARS
            if start < 0 or end == text_len:
                if end == text_len:
                    break
                else:
                    start = end # fail-safe

        return chunks

    @staticmethod
    def _get_file_hash(file_path: Path) -> str:
        """Returns SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def parse_txt(file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
    def parse_md(file_path: Path) -> str:
        # Same as TXT
        return DocumentParser.parse_txt(file_path)

    @staticmethod
    def parse_csv(file_path: Path) -> str:
        text = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                text.append(", ".join(row))
        return "\n".join(text)

    @staticmethod
    def parse_pdf(file_path: Path) -> str:
        if PdfReader is None:
            raise ImportError("pypdf is not installed.")
        reader = PdfReader(str(file_path))
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "\n".join(text)

    @staticmethod
    def parse_docx(file_path: Path) -> str:
        if DocxDocument is None:
            raise ImportError("python-docx is not installed.")
        doc = DocxDocument(str(file_path))
        return "\n".join([p.text for p in doc.paragraphs if p.text])

    @classmethod
    def process_file(cls, file_path_str: str) -> Dict[str, Any]:
        """
        Parses a file, hashes it, chunkifies text.
        Returns metadata and chunks.
        """
        file_path = Path(file_path_str)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        
        file_hash = cls._get_file_hash(file_path)
        
        text = ""
        if ext == ".txt":
            text = cls.parse_txt(file_path)
        elif ext == ".md":
            text = cls.parse_md(file_path)
        elif ext == ".csv":
            text = cls.parse_csv(file_path)
        elif ext == ".pdf":
            text = cls.parse_pdf(file_path)
        elif ext == ".docx":
            text = cls.parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks = cls._chunk_text(text)
        
        return {
            "filename": file_path.name,
            "filepath": str(file_path.absolute()),
            "filehash": file_hash,
            "extension": ext,
            "chunks": chunks,
        }
