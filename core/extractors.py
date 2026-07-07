"""extractors.py — Document parsing: PDF, EPUB, DOCX, TXT/MD → list of (page, text)."""
from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

from pypdf import PdfReader

from core.text_utils import clean_text, strip_markup


# ---------------------------------------------------------------------------
# DOCX extractor
# ---------------------------------------------------------------------------

def extract_docx_pages(path: Path) -> list[tuple[int, str]]:
    """Extract text from a DOCX file, returning (page_index, text) pairs."""
    pages: list[tuple[int, str]] = []
    with zipfile.ZipFile(path) as docx:
        names = set(docx.namelist())
        document_names = [
            name
            for name in names
            if name == "word/document.xml"
            or (
                name.startswith("word/")
                and name.endswith(".xml")
                and ("header" in name or "footer" in name)
            )
        ]
        document_names.sort(key=lambda name: 0 if name == "word/document.xml" else 1)
        for index, name in enumerate(document_names, start=1):
            xml = docx.read(name)
            root = ElementTree.fromstring(xml)
            parts: list[str] = []
            for element in root.iter():
                if element.tag.endswith("}t") and element.text:
                    parts.append(element.text)
                elif element.tag.endswith("}tab"):
                    parts.append(" ")
                elif element.tag.endswith("}br"):
                    parts.append("\n")
            text = clean_text(" ".join(parts))
            if text:
                pages.append((index, text))
    return pages


# ---------------------------------------------------------------------------
# EPUB extractor
# ---------------------------------------------------------------------------

def extract_epub_pages(path: Path) -> list[tuple[int, str]]:
    """Extract text from an EPUB file in spine order, returning (page_index, text) pairs."""
    pages: list[tuple[int, str]] = []
    with zipfile.ZipFile(path) as book:
        names = set(book.namelist())

        # Locate the OPF rootfile
        rootfile: str | None = None
        if "META-INF/container.xml" in names:
            container = book.read("META-INF/container.xml")
            tree = ElementTree.fromstring(container)
            for element in tree.iter():
                if element.tag.endswith("rootfile"):
                    rootfile = element.attrib.get("full-path")
                    break
        if not rootfile:
            candidates = [name for name in book.namelist() if name.lower().endswith(".opf")]
            rootfile = candidates[0] if candidates else None

        # Parse spine order from OPF
        ordered_docs: list[str] = []
        if rootfile and rootfile in names:
            base = str(Path(rootfile).parent).replace("\\", "/")
            if base == ".":
                base = ""
            package = ElementTree.fromstring(book.read(rootfile))
            manifest: dict[str, str] = {}
            spine: list[str] = []
            for element in package.iter():
                tag = element.tag.split("}", 1)[-1]
                if tag == "item":
                    item_id = element.attrib.get("id")
                    href = element.attrib.get("href")
                    media_type = element.attrib.get("media-type", "")
                    if item_id and href and (
                        "html" in media_type
                        or href.lower().endswith((".html", ".xhtml", ".htm"))
                    ):
                        manifest[item_id] = (base + "/" + href).lstrip("/") if base else href
                elif tag == "itemref":
                    itemref = element.attrib.get("idref")
                    if itemref:
                        spine.append(itemref)
            ordered_docs = [manifest[item_id] for item_id in spine if item_id in manifest]

        if not ordered_docs:
            ordered_docs = [
                name
                for name in book.namelist()
                if name.lower().endswith((".html", ".xhtml", ".htm"))
            ]

        seen: set[str] = set()
        for index, name in enumerate(ordered_docs, start=1):
            if name in seen or name not in names:
                continue
            seen.add(name)
            raw = book.read(name).decode("utf-8", errors="ignore")
            text = strip_markup(raw)
            if text:
                pages.append((index, text))
    return pages


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) tuples for any supported document format."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages: list[tuple[int, str]] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = clean_text(page.extract_text() or "")
            except Exception:
                text = ""
            if text:
                pages.append((index, text))
        if not pages:
            raise ValueError(
                "No readable text was found in this PDF. "
                "If it is scanned, run OCR first or upload a text-based PDF."
            )
        return pages

    if suffix == ".epub":
        pages = extract_epub_pages(path)
        if not pages:
            raise ValueError("No readable text was found in this EPUB file.")
        return pages

    if suffix == ".docx":
        pages = extract_docx_pages(path)
        if not pages:
            raise ValueError("No readable text was found in this DOCX file.")
        return pages

    if suffix == ".doc":
        raise ValueError(
            "Legacy .doc files are not readable yet. "
            "Save the document as .docx, PDF, EPUB, TXT, or MD and upload that file."
        )

    if suffix in {".txt", ".md"}:
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            raise ValueError("No readable text was found in this text file.")
        return [(1, text)]

    raise ValueError("RAG library accepts PDF, EPUB, DOCX, TXT, and MD files.")
