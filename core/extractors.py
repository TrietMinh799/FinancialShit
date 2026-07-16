"""extractors.py — Document parsing: PDF, EPUB, DOCX, TXT/MD → list of (page, text)."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from core.text_utils import clean_text

# lxml is optional but ~5-10x faster for XML/HTML parsing
try:
    from lxml import etree as _lxml_etree
    LXML_AVAILABLE = True
except Exception:
    _lxml_etree = None
    LXML_AVAILABLE = False

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


def _iter_xml_text(data: bytes) -> list[str]:
    """Walk XML *data* and collect all ``<w:t>`` text nodes.

    Uses lxml's fast C iterparse when available; falls back to stdlib
    ``ElementTree.iterparse`` which is slower but functionally equivalent.
    """
    parts: list[str] = []
    tag = f"{{{_NS_W}}}t"
    if _lxml_etree is not None:
        for _, elem in _lxml_etree.iterparse(data, tag=tag, events=("end",)):
            if elem.text:
                parts.append(elem.text)
            elem.clear()
    else:
        import xml.etree.ElementTree as _ET

        for _, elem in _ET.iterparse(data, events=("end",)):
            if elem.tag == tag and elem.text:
                parts.append(elem.text)
            elem.clear()
    return parts


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
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="ignore")
    return clean_text(re.sub(r"(?s)<[^>]+>", " ", data))


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
            media_type: str | None = item.get("media-type") or ""
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
            media_type = child.get("media-type", "")
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
# DOCX extractor
# ---------------------------------------------------------------------------


def extract_docx_pages(path: Path) -> list[tuple[int, str]]:
    """Extract text from a DOCX file, returning (page_index, text) pairs."""
    pages: list[tuple[int, str]] = []
    with zipfile.ZipFile(path) as docx:
        names = set(docx.namelist())
        docs = [
            n
            for n in names
            if n == "word/document.xml"
            or (
                n.startswith("word/")
                and n.endswith(".xml")
                and ("header" in n or "footer" in n)
            )
        ]
        docs.sort(key=lambda n: 0 if n == "word/document.xml" else 1)

        for index, name in enumerate(docs, start=1):
            parts = _iter_xml_text(docx.read(name))
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

        # Locate OPF rootfile from container.xml
        rootfile: str | None = None
        if "META-INF/container.xml" in names:
            root = _xml_parse(book.read("META-INF/container.xml"))
            for child in root.iter():
                if child.tag.endswith("rootfile"):
                    rootfile = child.get("full-path")
                    break

        if not rootfile:
            candidates = [n for n in book.namelist() if n.lower().endswith(".opf")]
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
                for n in book.namelist()
                if n.lower().endswith((".html", ".xhtml", ".htm"))
            ]

        seen: set[str] = set()
        for index, name in enumerate(ordered, start=1):
            if name in seen or name not in names:
                continue
            seen.add(name)
            text = _html_text(book.read(name))
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
