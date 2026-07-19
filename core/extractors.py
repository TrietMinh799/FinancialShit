"""extractors.py — Document parsing: PDF, EPUB, DOCX, TXT/MD → (page, text) pairs."""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from core.text_utils import clean_text
import contextlib

# lxml is optional but ~5-10x faster for XML/HTML parsing
try:
    from lxml import etree as _lxml_etree  # type: ignore[reportAttributeAccessIssue]
    LXML_AVAILABLE = True
except Exception:
    _lxml_etree = None
    LXML_AVAILABLE = False

# pytesseract is optional; needed only for scanned PDFs
try:
    import pytesseract as _pytesseract

    _OCR_AVAILABLE = True
except Exception:
    _pytesseract = None  # type: ignore[assignment]
    _OCR_AVAILABLE = False

try:
    from PIL import Image as _PILImage

    _PIL_AVAILABLE = True
except Exception:
    _PILImage = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# ---------------------------------------------------------------------------
# XML / HTML helpers (unified wrappers so Pylance sees consistent signatures)
# ---------------------------------------------------------------------------


def _xml_parse(data: bytes) -> Any:
    """Parse *data* as XML and return the root element.

    Uses lxml when available, otherwise falls back to stdlib
    ``xml.etree.ElementTree``.
    """
    if _lxml_etree is not None:
        return _lxml_etree.fromstring(data)
    import xml.etree.ElementTree as _ET

    return _ET.fromstring(data)


def _html_text(data: str | bytes) -> str:
    """Extract visible text from an HTML document, stripping markup.

    Uses lxml's fast HTML parser when available; falls back to a
    simple regex tag-stripper on decoding errors or missing lxml.
    """
    if _lxml_etree is not None:
        try:
            from lxml.html import fromstring as _fromstring

            doc = _fromstring(data)
            return clean_text(doc.text_content())
        except Exception:
            pass
    text_str: str = data if isinstance(data, str) else bytes(data).decode("utf-8", errors="ignore")
    return clean_text(re.sub(r"(?s)<[^>]+>", " ", text_str))


# ---------------------------------------------------------------------------
# OPF helpers (EPUB package document)
# ---------------------------------------------------------------------------


def _parse_opf_manifest(
    root: Any,
    base: str,
) -> dict[str, str]:
    """Extract ``id -> href`` mapping from an OPF ``<manifest>``.

    Only entries whose ``media-type`` contains ``html`` or whose ``href``
    ends with ``.html`` / ``.xhtml`` / ``.htm`` are included.
    """
    manifest: dict[str, str] = {}

    if _lxml_etree is not None:
        items = root.xpath("//*[local-name()='item']")
        for item in items:
            item_id: str | None = item.get("id")
            href: str | None = item.get("href")
            media_type: str = item.get("media-type") or ""
            if not item_id or not href:
                continue
            if "html" in media_type or href.lower().endswith((".html", ".xhtml", ".htm")):
                manifest[item_id] = f"{base}/{href}".lstrip("/") if base else href
    else:
        for child in root.iter():
            tag = child.tag.split("}", 1)[-1]
            if tag != "item":
                continue
            item_id = child.get("id")
            href = child.get("href")
            media_type: str = child.get("media-type") or ""
            if not item_id or not href:
                continue
            if "html" in media_type or href.lower().endswith((".html", ".xhtml", ".htm")):
                manifest[item_id] = f"{base}/{href}".lstrip("/") if base else href

    return manifest


def _parse_opf_spine(root: Any) -> list[str]:
    """Extract the ordered list of ``idref`` values from an OPF ``<spine>``."""
    spine: list[str] = []

    if _lxml_etree is not None:
        refs = root.xpath("//*[local-name()='itemref']")
        spine = [r.get("idref") for r in refs if r.get("idref")]
    else:
        for child in root.iter():
            tag = child.tag.split("}", 1)[-1]
            if tag == "itemref":
                idref = child.get("idref")
                if idref:
                    spine.append(idref)

    return spine


# ---------------------------------------------------------------------------
# Table → markdown helper
# ---------------------------------------------------------------------------


def _table_to_markdown(rows: list[list[str]]) -> str:
    """Convert a table (list of rows of cell strings) to a markdown table.

    Returns an empty string when the table has no usable content.
    """
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [clean_text(str(c)) if c is not None else "" for c in row]
        if any(cells):
            cleaned.append(cells)
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    lines = []
    for i, row in enumerate(cleaned):
        row = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("|" + "---|" * width)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DOCX extractor (python-docx)
# ---------------------------------------------------------------------------


def _safe_zip_entry(name: str) -> bool:
    """Reject zip entry names with path traversal components."""
    parts = name.replace("\\", "/").split("/")
    return not any(p in ("..", "") for p in parts)


# Target characters per pseudo-page for formats without real pages (DOCX).
_PSEUDO_PAGE_CHARS = 4000


def extract_docx_pages(path: Path) -> Iterator[tuple[int, str]]:
    """Extract text from a DOCX file, yielding (page_index, text) pairs.

    Uses python-docx to walk body paragraphs and tables in document order.
    Tables are rendered as markdown so their structure survives chunking.
    Headers and footers are deliberately excluded (boilerplate noise).
    DOCX has no true pages, so content is grouped into ~4000-char
    pseudo-pages to keep page metadata meaningful for citations.
    """
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = Document(str(path))
    except Exception as exc:
        raise ValueError(
            "Could not open this DOCX file. It may be corrupt or "
            "password-protected. Remove the password or re-save it and try again."
        ) from exc

    # Collect text blocks (paragraphs + markdown tables) in document order
    blocks: list[str] = []
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            text = clean_text(Paragraph(child, document).text)
            if text:
                blocks.append(text)
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            md = _table_to_markdown(rows)
            if md:
                blocks.append(md)

    # Group blocks into pseudo-pages and yield
    current: list[str] = []
    current_len = 0
    page_num = 0
    for block in blocks:
        current.append(block)
        current_len += len(block)
        if current_len >= _PSEUDO_PAGE_CHARS:
            page_num += 1
            yield (page_num, "\n\n".join(current))
            current = []
            current_len = 0
    if current:
        page_num += 1
        yield (page_num, "\n\n".join(current))


# ---------------------------------------------------------------------------
# EPUB extractor
# ---------------------------------------------------------------------------


def extract_epub_pages(path: Path) -> Iterator[tuple[int, str]]:
    """Extract text from an EPUB file in spine order, yielding (page_index, text) pairs."""
    with zipfile.ZipFile(path) as book:
        all_names = book.namelist()
        names = set(n for n in all_names if _safe_zip_entry(n))

        # Locate OPF rootfile from container.xml
        rootfile: str | None = None
        if "META-INF/container.xml" in names:
            root = _xml_parse(book.read("META-INF/container.xml"))
            for child in root.iter():
                if child.tag.endswith("rootfile"):
                    rootfile = child.get("full-path")
                    break

        if not rootfile:
            candidates = [n for n in all_names if n.lower().endswith(".opf") and _safe_zip_entry(n)]
            rootfile = candidates[0] if candidates else None

        # Read OPF and resolve spine-ordered HTML docs
        ordered: list[str] = []
        if rootfile and rootfile in names:
            base = str(Path(rootfile).parent).replace("\\", "/")
            if base == ".":
                base = ""
            opf_root = _xml_parse(book.read(rootfile))
            manifest = _parse_opf_manifest(opf_root, base)
            spine = _parse_opf_spine(opf_root)
            ordered = [manifest[s] for s in spine if s in manifest]

        if not ordered:
            ordered = [
                n
                for n in all_names
                if n.lower().endswith((".html", ".xhtml", ".htm")) and _safe_zip_entry(n)
            ]

        seen: set[str] = set()
        for index, name in enumerate(ordered, start=1):
            if name in seen or name not in names:
                continue
            seen.add(name)
            text = _html_text(book.read(name))
            if text:
                yield (index, text)


# ---------------------------------------------------------------------------
# OCR helper (scanned PDFs)
# ---------------------------------------------------------------------------


def _ocr_page(page: Any) -> str:
    """Render a pdfplumber page to an image and run OCR, returning the text.

    Requires ``pytesseract`` and ``Pillow`` to be installed, and the
    Tesseract-OCR engine to be on the system PATH.
    """
    if not _OCR_AVAILABLE or not _PIL_AVAILABLE:
        return ""
    try:
        import subprocess, sys

        subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return ""
    try:
        img = page.to_image(resolution=250)
        pil_image = img.original.convert("L")
        text = _pytesseract.image_to_string(pil_image)
        return clean_text(text or "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def extract_pages(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (page_number, text) tuples for any supported document format.

    All formats are processed lazily so large files stream without loading
    the entire document into memory.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        import pdfplumber  # type: ignore[reportMissingImports]

        pypdf_reader: PdfReader | None = None
        had_any_page = False

        try:
            pdf_handle = pdfplumber.open(str(path))
        except Exception as exc:
            raise ValueError(
                "Could not open this PDF. It may be corrupt, password-protected, "
                "or in an unsupported format. Remove the password and try again."
            ) from exc

        with pdf_handle as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                try:
                    text = clean_text(page.extract_text() or "")
                except Exception:
                    text = ""

                # Per-page pypdf fallback when pdfplumber gets no text
                if not text:
                    if pypdf_reader is None:
                        with contextlib.suppress(Exception):
                            pypdf_reader = PdfReader(str(path))
                    if pypdf_reader is not None:
                        with contextlib.suppress(Exception):
                            text = clean_text(
                                pypdf_reader.pages[index - 1].extract_text() or ""
                            )

                # Extract tables as markdown and append to text
                try:
                    tables = page.extract_tables()
                    if tables:
                        md_tables: list[str] = []
                        for tbl in tables:
                            md = _table_to_markdown(tbl)
                            if md:
                                md_tables.append(md)
                        if md_tables:
                            table_section = "\n\n" + "\n\n".join(md_tables)
                            text = (text + table_section) if text else table_section
                            text = clean_text(text)
                except Exception:
                    pass

                # OCR fallback for scanned pages
                if not text:
                    text = _ocr_page(page)

                if text:
                    had_any_page = True
                    yield (index, text)

        # OCR full-document fallback for completely scanned PDFs
        if not had_any_page:
            try:
                with pdfplumber.open(str(path)) as pdf:
                    for index, page in enumerate(pdf.pages, start=1):
                        text = _ocr_page(page)
                        if text:
                            had_any_page = True
                            yield (index, text)
            except Exception:
                pass

        if not had_any_page:
            raise ValueError(
                "No readable text was found in this PDF. "
                "If it is scanned, install Tesseract OCR (https://github.com/tesseract-ocr/tesseract)"
                "and ensure it is on your PATH, then try again."
            )
        return

    if suffix == ".epub":
        had_any_page = False
        for page in extract_epub_pages(path):
            had_any_page = True
            yield page
        if not had_any_page:
            raise ValueError("No readable text was found in this EPUB file.")
        return

    if suffix == ".docx":
        had_any_page = False
        for page in extract_docx_pages(path):
            had_any_page = True
            yield page
        if not had_any_page:
            raise ValueError("No readable text was found in this DOCX file.")
        return

    if suffix == ".doc":
        raise ValueError(
            "Legacy .doc files are not readable yet. "
            "Save the document as .docx, PDF, EPUB, TXT, or MD and upload that file."
        )

    if suffix in {".txt", ".md"}:
        raw = path.read_bytes()
        text = clean_text(raw.decode("utf-8", errors="replace"))
        if "\ufffd" in text:
            import logging as _log

            _log.getLogger("rag.extractors").warning(
                "Encoding issues detected in %s; replaced unreadable bytes with \ufffd. "
                "If the output looks garbled, try re-saving the file as UTF-8.",
                path.name,
            )
        if not text:
            raise ValueError("No readable text was found in this text file.")

        # Split into pseudo-pages on paragraph boundaries for streaming
        paragraphs = text.split("\n\n")
        page_num = 0
        buffer: list[str] = []
        buffer_len = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            buffer.append(para)
            buffer_len += len(para)
            if buffer_len >= _PSEUDO_PAGE_CHARS:
                page_num += 1
                yield (page_num, "\n\n".join(buffer))
                buffer = []
                buffer_len = 0
        if buffer:
            page_num += 1
            yield (page_num, "\n\n".join(buffer))
        return

    raise ValueError("RAG library accepts PDF, EPUB, DOCX, TXT, and MD files.")
