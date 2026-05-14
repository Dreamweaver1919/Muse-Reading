from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import eval.run_eval as eval_module
from backend.models import QuestionRequest
from services.ingest.parser import build_book_record
from services.ingest import parser as ingest_parser
from services.qa import answering as qa_answering
from services.summary import chapter_summary as summary_module


def demo_record():
    source = Path(__file__).resolve().parents[1] / "examples" / "muse_demo_book.txt"
    return build_book_record("muse_demo_book", source.read_text(encoding="utf-8"), source)


def build_test_epub_bytes() -> bytes:
    epub_path = Path(__file__).with_name("tmp_test_book.epub")
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


def test_ingestion_builds_multiple_chapters():
    record = demo_record()
    assert record.chapter_count == 3
    assert len(record.chunks) >= 6


def test_spoiler_question_is_blocked(monkeypatch):
    record = demo_record()

    def fake_generate(**kwargs):
        assert "拒绝剧透" in kwargs["question"]
        return "我不能直接透露后文，但可以先回到你当前看到的这段。", "qa-model", []

    monkeypatch.setattr(qa_answering, "generate_persona_response", fake_generate)
    response = qa_answering.build_answer(
        QuestionRequest(
            book_id=record.book_id,
            question="What is Aya's ending?",
            highlight_text=record.chunks[0].text,
            current_chapter=1,
        ),
        record.chunks,
    )
    assert response.safe is False
    assert response.reason == "question_requests_future_plot"
    assert response.model_name == "qa-model"


def test_summary_returns_current_chapter_only(monkeypatch):
    record = demo_record()

    def fake_generate(**kwargs):
        visible_contexts = kwargs["visible_contexts"]
        assert any("Lin opened the old notebook" in item for item in visible_contexts)
        assert all("Aya returned the next afternoon" not in item for item in visible_contexts)
        return "这是基于第一章已读内容的总结。", "summary-model", []

    monkeypatch.setattr(summary_module, "generate_persona_response", fake_generate)
    summary = summary_module.summarize_chapter(record, 1, "neutral")
    assert summary.chapter_id == "chapter-001"
    assert summary.summary == "这是基于第一章已读内容的总结。"
    assert summary.model_name == "summary-model"


def test_eval_runner_reports_demo_sections():
    def fake_eval_answer(request, chunks):
        from backend.models import QuestionResponse, RetrievedContext

        chunk = chunks[0]
        return QuestionResponse(
            answer="测试回答",
            persona_id=request.persona_id,
            safe=request.question.lower() not in {"what is aya's ending?", "who dies later?"},
            reason="within_visible_scope",
            contexts=[
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    chapter_index=chunk.chapter_index,
                    paragraph_index=chunk.paragraph_index,
                    score=1.0,
                    text=chunk.text,
                )
            ],
            model_name="eval-model",
        )

    def fake_eval_summary(book, current_chapter, persona_id):
        from backend.models import SummaryResponse

        return SummaryResponse(
            summary="Lin opened the old notebook and Aya returned the next afternoon.",
            chapter_id=f"chapter-{current_chapter:03d}",
            persona_id=persona_id,
            model_name="eval-summary-model",
        )

    original_answer = eval_module.build_answer
    original_summary = eval_module.summarize_chapter
    eval_module.build_answer = fake_eval_answer
    eval_module.summarize_chapter = fake_eval_summary
    try:
        result = eval_module.run_evaluation()
    finally:
        eval_module.build_answer = original_answer
        eval_module.summarize_chapter = original_summary
    assert result["book_id"] == "muse-demo-book"
    assert result["overall"]["failed"] >= 0
    assert result["highlight_qa"]["sample_count"] >= 1
    assert result["anti_spoiler"]["sample_count"] >= 1
    assert result["chapter_summary"]["sample_count"] >= 1


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


def test_qa_uses_persona_generation_with_visible_context(monkeypatch):
    record = demo_record()

    def fake_generate(**kwargs):
        assert kwargs["task"] == "qa"
        assert kwargs["persona_id"] == "lu-xun"
        assert kwargs["current_chapter"] == 1
        assert kwargs["highlight_text"]
        assert kwargs["visible_contexts"]
        return "这段文字先写动作，再压出人物之间的紧张。", "qa-model", []

    monkeypatch.setattr(qa_answering, "generate_persona_response", fake_generate)
    response = qa_answering.build_answer(
        QuestionRequest(
            book_id=record.book_id,
            question="这段在写什么？",
            highlight_text=record.chunks[0].text,
            current_chapter=1,
            persona_id="lu-xun",
        ),
        record.chunks,
    )
    assert response.safe is True
    assert response.answer == "这段文字先写动作，再压出人物之间的紧张。"
    assert response.model_name == "qa-model"
