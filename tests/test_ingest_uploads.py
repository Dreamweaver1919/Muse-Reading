from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from services.ingest import parser as ingest_parser
from services.ingest.parser import build_book_record, build_book_record_from_upload


def build_test_epub_bytes() -> bytes:
    epub_path = Path(__file__).with_name("tmp_ingest_test_book.epub")
    with ZipFile(epub_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0" encoding="utf-8"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>""",
        )
        archive.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="utf-8"?>
            <package version="2.0" xmlns="http://www.idpf.org/2007/opf">
              <manifest>
                <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
                <item id="chapter2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine>
                <itemref idref="chapter1"/>
                <itemref idref="chapter2"/>
              </spine>
            </package>""",
        )
        archive.writestr(
            "OEBPS/chapter1.xhtml",
            "<html><body><h1>Chapter 1</h1><p>Lin opens the archive.</p></body></html>",
        )
        archive.writestr(
            "OEBPS/chapter2.xhtml",
            "<html><body><h1>Chapter 2</h1><p>Aya studies the second clue.</p></body></html>",
        )
    raw_bytes = epub_path.read_bytes()
    epub_path.unlink(missing_ok=True)
    return raw_bytes


def build_epub_with_front_matter_bytes() -> bytes:
    epub_path = Path(__file__).with_name("tmp_ingest_front_matter.epub")
    with ZipFile(epub_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0" encoding="utf-8"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>""",
        )
        archive.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="utf-8"?>
            <package version="2.0" xmlns="http://www.idpf.org/2007/opf">
              <manifest>
                <item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>
                <item id="toc" href="toc.xhtml" media-type="application/xhtml+xml"/>
                <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
                <item id="chapter2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine>
                <itemref idref="cover"/>
                <itemref idref="toc"/>
                <itemref idref="chapter1"/>
                <itemref idref="chapter2"/>
              </spine>
            </package>""",
        )
        archive.writestr(
            "OEBPS/cover.xhtml",
            "<html><body><h1>Cover</h1><p>Sample Book</p></body></html>",
        )
        archive.writestr(
            "OEBPS/toc.xhtml",
            "<html><body><h1>目录</h1><p>第1章</p><p>第2章</p></body></html>",
        )
        archive.writestr(
            "OEBPS/chapter1.xhtml",
            "<html><body><p>多年以后，面对行刑队，奥雷里亚诺会想起那个下午。</p><p>马孔多还是一座刚刚诞生的村落。</p></body></html>",
        )
        archive.writestr(
            "OEBPS/chapter2.xhtml",
            "<html><body><p>乌尔苏拉拒绝向命运低头，她继续经营家业。</p><p>孩子们在院子里长大。</p></body></html>",
        )
    raw_bytes = epub_path.read_bytes()
    epub_path.unlink(missing_ok=True)
    return raw_bytes


def test_parser_extracts_plain_text():
    text = ingest_parser.read_uploaded_text("notes.txt", b"Chapter 1\n\nLin opens the archive.")
    assert "Lin opens the archive." in text


def test_parser_extracts_epub_text_into_chapters():
    text = ingest_parser.read_uploaded_text("sample.epub", build_test_epub_bytes())
    assert "Chapter 1" in text
    assert "Lin opens the archive." in text
    assert "Chapter 2" in text


def test_parser_extracts_pdf_text_with_local_reader(monkeypatch):
    class FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakeReader:
        def __init__(self, _stream):
            self.pages = [FakePage("Chapter 1\nLin enters the square."), FakePage("Chapter 2\nAya responds.")]

    monkeypatch.setattr(ingest_parser, "PdfReader", FakeReader)
    text = ingest_parser.read_uploaded_text("sample.pdf", b"%PDF-test")
    assert "Lin enters the square." in text
    assert "Chapter 2" in text


def test_epub_text_flows_into_existing_book_record_builder():
    text = ingest_parser.read_uploaded_text("reading-demo.epub", build_test_epub_bytes())
    record = build_book_record("reading-demo", text, Path("reading-demo.epub"))
    assert record.book_id == "reading-demo"
    assert record.chapter_count == 2
    assert len(record.chunks) >= 2


def test_epub_upload_builder_skips_front_matter_toc_and_keeps_real_sections():
    record = build_book_record_from_upload(
        title="front-matter-demo",
        filename="front-matter-demo.epub",
        raw_bytes=build_epub_with_front_matter_bytes(),
        source_path=Path("front-matter-demo.epub"),
    )
    assert record.chapter_count == 2
    assert len(record.chunks) >= 2
    assert "多年以后" in record.chunks[0].text


def test_build_book_record_merges_hard_wrapped_paragraph_fragments():
    raw_text = (
        "Chapter 1\n\n"
        "William Somerset Maugham, a celebrated English writer\n\n"
        "began his long writing career with Liza of Lambeth.\n\n"
        "He wrote novels, short stories, and essays."
    )
    record = build_book_record("wrapped-demo", raw_text, Path("wrapped-demo.txt"))
    assert record.chapter_count == 1
    assert len(record.chunks) == 2
    assert "William Somerset Maugham, a celebrated English writer began his long writing career" in record.chunks[0].text
