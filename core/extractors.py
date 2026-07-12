"""extractors.py — Document parsing: PDF, EPUB, DOCX, TXT/MD → list of (page, text)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from pypdf import PdfReader

from core.text_utils import clean_text

# lxml is optional but ~5-10x faster for XML/HTML parsing
try:
    from lxml import etree as LET
    from lxml.html import fromstring as html_fromstring
    LXML_AVAILABLE = True
except Exception:  # pragma: no cover
    from xml.etree import ElementTree as LET
    LXML_AVAILABLE = False
    def html_fromstring(data: str):
        """Fallback: strip tags with regex (slow)."""
        import re
        return re.sub(r"(?s)<[^>]+>", " ", data)

# ---------------------------------------------------------------------------
# DOCX extractor (optimized with lxml iterparse / XPath)
# ---------------------------------------------------------------------------


def extract_docx_pages(path: Path) -> list[tuple[int, str]]:
    """Extract text from a DOCX file, returning (page_index, text) pairs."""
    pages: list[tuple[int, str]] = []
    with zipfile.ZipFile(path) as docx:
        # Only read document.xml + headers/footers
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

        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

        for index, name in enumerate(document_names, start=1):
            xml_bytes = docx.read(name)
            if LXML_AVAILABLE:
                # Streaming parse: only visit <w:t> text nodes
                parts = []
                for _, elem in LET.iterparse(xml_bytes, tag="{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t", events=("end",)):
                    if elem.text:
                        parts.append(elem.text)
                    elem.clear()  # free memory
                text = clean_text(" ".join(parts))
            else:
                # Fallback: stdlib ElementTree (slower)
                root = LET.fromstring(xml_bytes)
                parts = [
                    elem.text for elem in root.iter()
                    if elem.tag.endswith("}t") and elem.text
                ]
                text = clean_text(" ".join(parts))

            if text:
                pages.append((index, text))
    return pages


# ---------------------------------------------------------------------------
# EPUB extractor (optimized with lxml.html + streaming)
# ---------------------------------------------------------------------------


def extract_epub_pages(path: Path) -> list[tuple[int, str]]:
    """Extract text from an EPUB file in spine order, returning (page_index, text) pairs."""
    pages: list[tuple[int, str]] = []
    with zipfile.ZipFile(path) as book:
        names = set(book.namelist())

        # Locate OPF rootfile
        rootfile: str | None = None
        if "META-INF/container.xml" in names:
            container_bytes = book.read("META-INF/container.xml")
            if LXML_AVAILABLE:
                tree = LET.fromstring(container_bytes)
                rootfile_elem = tree.find(".//{*}rootfile")
                if rootfile_elem is not None:
                    rootfile = rootfile_elem.get("full-path")
            else:
                import xml.etree.ElementTree as ET
                tree = ET.fromstring(container_bytes)
                for elem in tree.iter():
                    if elem.tag.endswith("rootfile"):
                        rootfile = elem.attrib.get("full-path")
                        break

        if not rootfile:
            candidates = [n for n in book.namelist() if n.lower().endswith(".opf")]
            rootfile = candidates[0] if candidates else None

        ordered_docs: list[str] = []
        if rootfile and rootfile in names:
            base = str(Path(rootfile).parent).replace("\\", "/")
            if base == ".":
                base = ""
            opf_bytes = book.read(rootfile)
            if LXML_AVAILABLE:
                pkg = LET.fromstring(opf_bytes)
                # Manifest: id -> href (HTML only)
                manifest = {
                    item.get("id"): (base + "/" + item.get("href")).lstrip("/") if base else item.get("href")
                    for item in pkg.xpath("//*[local-name()='item']")
                    if item.get("id") and item.get("href")
                    and ("html" in (item.get("media-type", "") or "")
                         or item.get("href", "").lower().endswith((".html", ".xhtml", ".htm")))
                }
                # Spine order
                spine_ids = [
                    itemref.get("idref")
                    for itemref in pkg.xpath("//*[local-name()='itemref']")
                    if itemref.get("idref")
                ]
                ordered_docs = [manifest[sid] for sid in spine_ids if sid in manifest]
            else:
                # Fallback stdlib
                import xml.etree.ElementTree as ET
                pkg = ET.fromstring(opf_bytes)
                manifest = {}
                spine = []
                for elem in pkg.iter():
                    tag = elem.tag.split("}", 1)[-1]
                    if tag == "item":
                        item_id = elem.attrib.get("id")
                        href = elem.attrib.get("href")
                        media_type = elem.attrib.get("media-type", "")
                        if item_id and href and (
                            "html" in media_type
                            or href.lower().endswith((".html", ".xhtml", ".htm"))
                        ):
                            manifest[item_id] = (base + "/" + href).lstrip("/") if base else href
                    elif tag == "itemref":
                        itemref = elem.attrib.get("idref")
                        if itemref:
                            spine.append(itemref)
                ordered_docs = [manifest[sid] for sid in spine if sid in manifest]

        if not ordered_docs:
            ordered_docs = [
                n for n in book.namelist()
                if n.lower().endswith((".html", ".xhtml", ".htm"))
            ]

        seen: set[str] = set()
        for index, name in enumerate(ordered_docs, start=1):
            if name in seen or name not in names:
                continue
            seen.add(name)
            raw_bytes = book.read(name)
            # lxml.html handles encoding + extracts text efficiently
            if LXML_AVAILABLE:
                try:
                    doc = html_fromstring(raw_bytes)
                    text = doc.text_content()
                except Exception:
                    text = raw_bytes.decode("utf-8", errors="ignore")
                    import re
                    text = re.sub(r"(?s)<[^>]+>", " ", text)
            else:
                raw = raw_bytes.decode("utf-8", errors="ignore")
                import re
                text = re.sub(r"(?s)<[^>]+>", " ", raw)

            text = clean_text(text)
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
