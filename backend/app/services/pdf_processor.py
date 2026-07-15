import pymupdf
import re
from pathlib import Path
from typing import List, Tuple
import hashlib


class PDFProcessor:
    """Process PDF documents for RAG system"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        # Both measured in words, configured via settings (not hardcoded)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF"""
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    @staticmethod
    def _is_heading(line: str) -> bool:
        """Heuristic: does this line look like a section heading?"""
        line = line.strip()
        if not line or len(line) > 80 or line.endswith((".", ",", ";")):
            return False
        # "Answer 1:", "Question 3", "Section 2.1", "Chapter 4" style headings
        if re.match(r"^(answer|question|section|chapter|part|unit)\s*\d+", line, re.IGNORECASE):
            return True
        # ALL-CAPS lines like "NATIONAL INSTITUTE OF TECHNOLOGY JAIPUR"
        letters = [c for c in line if c.isalpha()]
        return len(letters) >= 4 and all(c.isupper() for c in letters)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split a paragraph into sentences on ., !, ? boundaries."""
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _split_paragraphs(self, text: str) -> List[Tuple[str, str]]:
        """
        Split raw text into (paragraph_text, section_title) pairs.
        A paragraph ends at a blank line; a heading line starts a new section
        (the heading itself stays in the text so its keywords remain searchable).
        """
        paragraphs = []
        section = ""
        current_lines: List[str] = []

        def flush():
            if current_lines:
                paragraphs.append((" ".join(current_lines), section))
                current_lines.clear()

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                flush()
                continue
            if self._is_heading(stripped):
                flush()
                section = stripped
            current_lines.append(stripped)
        flush()
        return paragraphs

    def chunk_text(self, text: str) -> List[Tuple[str, dict]]:
        """
        Structure-aware chunking with recursive fallback:
        paragraph -> sentence -> word.

        Whole paragraphs are packed into chunks up to chunk_size words.
        A paragraph too large for one chunk is split into sentences; a
        sentence longer than chunk_size is split into words as a last resort.
        Consecutive chunks overlap by ~chunk_overlap words (whole sentences)
        so no fact is stranded on a chunk boundary.
        Returns list of (chunk_text, metadata) tuples.
        """
        # Build atomic units (never split below these, except monster sentences)
        units: List[Tuple[str, str]] = []  # (unit_text, section)
        for para, section in self._split_paragraphs(text):
            if len(para.split()) <= self.chunk_size:
                units.append((para, section))
            else:
                for sentence in self._split_sentences(para):
                    words = sentence.split()
                    if len(words) <= self.chunk_size:
                        units.append((sentence, section))
                    else:
                        # Word-level fallback for pathological sentences
                        for i in range(0, len(words), self.chunk_size):
                            units.append((" ".join(words[i:i + self.chunk_size]), section))

        # Greedily pack units into chunks, carrying sentence overlap between chunks
        chunks: List[Tuple[str, dict]] = []
        current: List[Tuple[str, str]] = []
        current_words = 0
        has_new_content = False  # does `current` hold anything beyond carried overlap?

        def flush_chunk():
            nonlocal current, current_words, has_new_content
            if not current:
                return
            chunk_body = "\n".join(u for u, _ in current)
            chunks.append((chunk_body, {
                "chunk_id": len(chunks),
                "section": current[0][1],
                "word_count": current_words,
            }))
            # Carry trailing units (~chunk_overlap words) into the next chunk
            overlap_units: List[Tuple[str, str]] = []
            overlap_words = 0
            for unit in reversed(current):
                unit_len = len(unit[0].split())
                if overlap_words + unit_len > self.chunk_overlap:
                    break
                overlap_units.insert(0, unit)
                overlap_words += unit_len
            current = overlap_units
            current_words = overlap_words
            has_new_content = False

        for unit_text, section in units:
            unit_len = len(unit_text.split())
            if current and current_words + unit_len > self.chunk_size:
                flush_chunk()
            current.append((unit_text, section))
            current_words += unit_len
            has_new_content = True

        if has_new_content:
            flush_chunk()

        return chunks

    def process_pdf(self, pdf_path: Path) -> Tuple[str, List[Tuple[str, dict]]]:
        """
        Process PDF: extract text and chunk it.
        Returns (document_id, chunks).
        """
        # Generate document ID from file hash
        with open(pdf_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        # Extract and chunk text
        text = self.extract_text(pdf_path)
        chunks = self.chunk_text(text)

        return file_hash, chunks
