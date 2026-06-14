"""Tests for `app.ingestion.loaders`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.ingestion.loaders import load_document, load_pdf, load_text


def test_load_text_reads_utf8(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Hello, world!", encoding="utf-8")

    assert load_text(path) == "Hello, world!"


def test_load_document_dispatches_markdown(tmp_path: Path) -> None:
    path = tmp_path / "readme.md"
    path.write_text("# Title\n\nSome body text.", encoding="utf-8")

    assert load_document(path) == "# Title\n\nSome body text."


def test_load_document_dispatches_pdf(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as f:
        writer.write(f)

    text = load_document(path)
    assert isinstance(text, str)


def test_load_pdf_directly(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as f:
        writer.write(f)

    assert isinstance(load_pdf(path), str)


def test_load_document_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.docx"
    path.write_text("not really a docx", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)
