"""Loaders that extract raw text from supported document formats."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


def load_pdf(path: Path) -> str:
    """Extract and concatenate text from every page of a PDF file."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_text(path: Path) -> str:
    """Read a plain-text or Markdown file as UTF-8."""
    return path.read_text(encoding="utf-8")


def load_document(path: Path) -> str:
    """Dispatch to the correct loader based on file extension.

    Args:
        path: Path to a `.pdf`, `.md`/`.markdown`, or `.txt` file.

    Returns:
        The extracted plain text content of the document.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if suffix == ".pdf":
        return load_pdf(path)
    return load_text(path)
