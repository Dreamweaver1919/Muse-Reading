from __future__ import annotations

from io import BytesIO
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from backend.models import BookChunk, BookRecord

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - exercised in environments without optional deps.
    PdfReader = None


CHAPTER_PATTERNS = [
    re.compile(r"^\s*Chapter[\s_\-]*\d+.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*第\s*[0-9一二三四五六七八九十百千零〇两]+\s*章.*$", re.MULTILINE),
]

SUPPORTED_UPLOAD_SUFFIXES = {".txt", ".pdf", ".epub"}
EPUB_NAMESPACE = {"opf": "http://www.idpf.org/2007/opf"}
EPUB_FRONT_MATTER_HINTS = (
    "cover",
    "目录",
    "目次",
    "contents",
    "献给",
    "图书在版编目",
    "cip",
    "isbn",
    "版权所有",
    "版权",
    "责任编辑",
    "出版",
    "译",
    "著",
    "定价",
)


class UnsupportedUploadFormatError(ValueError):
    """Raised when a file cannot be ingested by the local parser."""


class UploadTextExtractionError(ValueError):
    """Raised when supported files do not yield readable text."""


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", lowered)
    return lowered.strip("-") or "book"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraph_blocks(text: str) -> list[str]:
    return [normalize_text(block) for block in text.split("\n\n") if normalize_text(block)]


def _looks_like_chapter_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern.match(stripped) for pattern in CHAPTER_PATTERNS)


def _ends_like_sentence(text: str) -> bool:
    return bool(re.search(r"[。！？!?\.\"'”’）)\]]\s*$", text))


def _join_text(left: str, right: str) -> str:
    if left.endswith("-"):
        return f"{left[:-1]}{right.lstrip()}"
    return f"{left.rstrip()} {right.lstrip()}".strip()


def merge_broken_paragraphs(paragraphs: list[str]) -> list[str]:
    merged: list[str] = []
    for paragraph in paragraphs:
        current = normalize_text(paragraph)
        if not current:
            continue
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        should_merge = (
            not _looks_like_chapter_heading(current)
            and not _looks_like_chapter_heading(previous)
            and not _ends_like_sentence(previous)
        )
        if should_merge:
            merged[-1] = _join_text(previous, current)
        else:
            merged.append(current)
    return merged


def read_uploaded_text(filename: str, raw_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise UnsupportedUploadFormatError(
            f"Unsupported upload format '{suffix or 'unknown'}'. Supported formats: txt, pdf, epub."
        )
    if suffix == ".txt":
        text = raw_bytes.decode("utf-8", errors="ignore")
    elif suffix == ".pdf":
        text = extract_pdf_text(raw_bytes)
    else:
        text = extract_epub_text(raw_bytes)

    normalized = normalize_text(text)
    if not normalized:
        raise UploadTextExtractionError(f"No readable text could be extracted from '{filename}'.")
    return normalized


def extract_pdf_text(raw_bytes: bytes) -> str:
    if PdfReader is None:
        raise UploadTextExtractionError("PDF support requires the optional 'pypdf' dependency.")

    reader = PdfReader(BytesIO(raw_bytes))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        cleaned = normalize_text(page_text)
        if cleaned:
            pages.append(cleaned)
    return "\n\n".join(pages)


def extract_epub_sections(raw_bytes: bytes) -> list[str]:
    try:
        archive = zipfile.ZipFile(BytesIO(raw_bytes))
    except zipfile.BadZipFile as exc:
        raise UploadTextExtractionError("The uploaded EPUB file is not a valid zip archive.") from exc

    with archive:
        container_path = "META-INF/container.xml"
        try:
            container_root = ElementTree.fromstring(archive.read(container_path))
        except KeyError as exc:
            raise UploadTextExtractionError("The EPUB container.xml file is missing.") from exc

        rootfile = container_root.find(".//{*}rootfile")
        if rootfile is None:
            raise UploadTextExtractionError("The EPUB container.xml file does not define a package document.")

        package_path = rootfile.attrib.get("full-path", "")
        if not package_path:
            raise UploadTextExtractionError("The EPUB package document path is empty.")

        try:
            package_root = ElementTree.fromstring(archive.read(package_path))
        except KeyError as exc:
            raise UploadTextExtractionError("The EPUB package document is missing.") from exc

        base_dir = Path(package_path).parent
        manifest_map = {
            item.attrib["id"]: item.attrib["href"]
            for item in package_root.findall(".//opf:manifest/opf:item", EPUB_NAMESPACE)
            if "id" in item.attrib and "href" in item.attrib
        }
        spine = package_root.findall(".//opf:spine/opf:itemref", EPUB_NAMESPACE)

        sections: list[str] = []
        for itemref in spine:
            item_id = itemref.attrib.get("idref")
            href = manifest_map.get(item_id or "")
            if not href:
                continue
            item_path = str((base_dir / href).as_posix())
            try:
                document = archive.read(item_path)
            except KeyError:
                continue
            section_text = extract_text_from_markup(document.decode("utf-8", errors="ignore"))
            if section_text:
                sections.append(section_text)
        return sections


def extract_epub_text(raw_bytes: bytes) -> str:
    return "\n\n".join(extract_epub_sections(raw_bytes))


def extract_text_from_markup(markup: str) -> str:
    without_scripts = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", markup, flags=re.IGNORECASE | re.DOTALL)
    with_breaks = re.sub(
        r"</?(p|div|section|article|h[1-6]|li|blockquote|tr|br)\b[^>]*>",
        "\n",
        without_scripts,
        flags=re.IGNORECASE,
    )
    text_only = re.sub(r"<[^>]+>", " ", with_breaks)
    text_only = (
        text_only.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
    )
    return normalize_text(text_only)


def split_chapters(text: str) -> list[tuple[str, str]]:
    normalized = normalize_text(text)
    for pattern in CHAPTER_PATTERNS:
        matches = list(pattern.finditer(normalized))
        if len(matches) >= 2:
            chapters: list[tuple[str, str]] = []
            for index, match in enumerate(matches):
                start = match.start()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
                title = match.group(0).strip()
                body = normalized[start:end].strip()
                chapters.append((title, body))
            return chapters
    return [("Chapter 1", normalized)]


def score_spoiler_level(chapter_index: int) -> int:
    if chapter_index <= 1:
        return 0
    if chapter_index <= 3:
        return 1
    return 2


def spoiler_label(spoiler_level: int) -> str:
    if spoiler_level <= 0:
        return "safe"
    if spoiler_level == 1:
        return "mild"
    return "high"


def extract_candidate_characters(paragraph: str) -> list[str]:
    matches = re.findall(r"\b[A-Z][a-z]{2,}\b", paragraph)
    blocked = {"Chapter", "When", "The", "By", "At"}
    deduped: list[str] = []
    for match in matches:
        if match in blocked:
            continue
        if match not in deduped:
            deduped.append(match)
    return deduped[:5]


def _build_book_record_from_chapter_bodies(
    title: str,
    chapter_bodies: list[tuple[str, list[str]]],
    source_path: Path,
) -> BookRecord:
    book_id = slugify(title)
    chunks: list[BookChunk] = []
    token_offset = 0
    for chapter_index, (chapter_title, paragraphs) in enumerate(chapter_bodies, start=1):
        for paragraph_index, paragraph in enumerate(paragraphs, start=1):
            chunk_id = f"{book_id}-c{chapter_index:03d}-p{paragraph_index:03d}"
            level = score_spoiler_level(chapter_index)
            chunks.append(
                BookChunk(
                    chunk_id=chunk_id,
                    book_id=book_id,
                    chapter_id=f"chapter-{chapter_index:03d}",
                    section_id=f"section-{chapter_index:03d}",
                    paragraph_start_id=f"paragraph-{paragraph_index:03d}",
                    paragraph_end_id=f"paragraph-{paragraph_index:03d}",
                    chunk_level="l0_raw_paragraph",
                    chapter_index=chapter_index,
                    paragraph_id=f"paragraph-{paragraph_index:03d}",
                    paragraph_index=paragraph_index,
                    text=paragraph,
                    token_offset=token_offset,
                    spoiler_level=level,
                    position={
                        "chapter_index": chapter_index,
                        "section_index": chapter_index,
                        "char_start": token_offset,
                        "char_end": token_offset + len(paragraph),
                    },
                    spoiler_guard={
                        "spoiler_level": spoiler_label(level),
                        "max_visible_chapter_index": chapter_index,
                    },
                    tags=["book_text", "demo_chunk"],
                    candidate_characters=extract_candidate_characters(paragraph),
                    metadata={"chapter_title": chapter_title},
                )
            )
            token_offset += max(1, len(paragraph.split()))
    return BookRecord(
        book_id=book_id,
        title=title,
        source_path=str(source_path),
        chapter_count=len(chapter_bodies),
        chunks=chunks,
    )


def _is_substantive_epub_section(paragraphs: list[str]) -> bool:
    if not paragraphs:
        return False

    combined = " ".join(paragraphs)
    sentence_hits = len(re.findall(r"[。！？!?\.]", combined))
    body_paragraphs = paragraphs[1:] if _looks_like_chapter_heading(paragraphs[0]) else paragraphs
    body_text = " ".join(body_paragraphs).strip()
    lowered = combined.lower()
    hint_hits = sum(1 for hint in EPUB_FRONT_MATTER_HINTS if hint in lowered)
    max_paragraph_length = max(len(paragraph) for paragraph in body_paragraphs) if body_paragraphs else 0

    if hint_hits >= 2 and max_paragraph_length < 200 and len(combined) < 2000:
        return False
    if hint_hits >= 1 and max_paragraph_length < 120 and len(combined) < 900:
        return False

    if len(combined) >= 400:
        return True
    if len(body_text) >= 40 and sentence_hits >= 1:
        return True
    if len(body_paragraphs) >= 2 and len(body_text) >= 60 and sentence_hits >= 1:
        return True
    if _looks_like_chapter_heading(paragraphs[0]) and len(body_text) >= 20:
        return True
    return False


def _epub_section_to_chapter(section_text: str, chapter_index: int) -> tuple[str, list[str]] | None:
    paragraphs = merge_broken_paragraphs(split_paragraph_blocks(section_text))
    if not paragraphs:
        return None
    while (
        len(paragraphs) > 1
        and len(paragraphs[0]) < 50
        and len(paragraphs[1]) > 80
        and not _looks_like_chapter_heading(paragraphs[0])
        and not re.search(r"[。！？!?\.]", paragraphs[0])
    ):
        paragraphs = paragraphs[1:]

    if _looks_like_chapter_heading(paragraphs[0]):
        chapter_title = paragraphs[0]
        body = paragraphs[1:]
    else:
        chapter_title = f"第{chapter_index}章"
        body = paragraphs

    cleaned_body = [paragraph for paragraph in body if paragraph and paragraph != chapter_title]
    if not cleaned_body:
        return None
    return chapter_title, cleaned_body


def build_epub_book_record(title: str, raw_bytes: bytes, source_path: Path) -> BookRecord:
    sections = extract_epub_sections(raw_bytes)
    chapter_bodies: list[tuple[str, list[str]]] = []
    started = False

    for section_text in sections:
        paragraphs = split_paragraph_blocks(section_text)
        if not paragraphs:
            continue
        is_substantive = _is_substantive_epub_section(paragraphs)
        if not started:
            if not is_substantive:
                continue
            started = True

        chapter = _epub_section_to_chapter(section_text, len(chapter_bodies) + 1)
        if chapter is None:
            continue
        chapter_bodies.append(chapter)

    if not chapter_bodies:
        text = extract_epub_text(raw_bytes)
        return build_book_record(title, text, source_path)
    return _build_book_record_from_chapter_bodies(title, chapter_bodies, source_path)


def build_book_record_from_upload(title: str, filename: str, raw_bytes: bytes, source_path: Path) -> BookRecord:
    suffix = Path(filename).suffix.lower()
    if suffix == ".epub":
        return build_epub_book_record(title, raw_bytes, source_path)
    text = read_uploaded_text(filename, raw_bytes)
    return build_book_record(title, text, source_path)


def build_book_record(title: str, raw_text: str, source_path: Path) -> BookRecord:
    chapters = split_chapters(raw_text)
    chapter_bodies: list[tuple[str, list[str]]] = []
    for chapter_title, chapter_text in chapters:
        paragraphs = merge_broken_paragraphs(split_paragraph_blocks(chapter_text))
        if paragraphs and paragraphs[0] == chapter_title.strip():
            paragraphs = paragraphs[1:]
        if paragraphs:
            chapter_bodies.append((chapter_title, paragraphs))
    return _build_book_record_from_chapter_bodies(title, chapter_bodies, source_path)
