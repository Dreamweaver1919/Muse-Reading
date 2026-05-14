from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi.testclient import TestClient

import api.app as app_module
from services.ingest import parser as ingest_parser


app = app_module.app


def build_test_epub_bytes() -> bytes:
    epub_path = Path(__file__).with_name("tmp_api_test_book.epub")
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


def test_graph_api_exposes_metadata_and_query():
    with TestClient(app) as client:
        metadata_response = client.get("/api/books/muse_demo_book/graph/metadata")
        assert metadata_response.status_code == 200
        metadata_payload = metadata_response.json()
        assert metadata_payload["graph_id"].startswith("graph::")
        assert metadata_payload["stats"]["chapter_count"] >= 2

        query_response = client.post(
            "/api/books/muse_demo_book/graph/query",
            json={
                "query": "Aya",
                "max_chapter": 2,
                "top_k": 4,
                "node_types": ["chapter", "episode", "entity"],
            },
        )
        assert query_response.status_code == 200
        query_payload = query_response.json()
        assert query_payload["visible_episode_count"] >= 1
        assert query_payload["graph_stats"]["chapter_count"] >= 2
        assert query_payload["hits"]


def test_upload_endpoint_accepts_epub_and_builds_graph():
    with TestClient(app) as client:
        response = client.post(
            "/api/upload",
            files={
                "file": (
                    "api_epub_upload.epub",
                    build_test_epub_bytes(),
                    "application/epub+zip",
                )
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["chapter_count"] == 2
        graph_response = client.get(f"/api/books/{payload['book_id']}/graph")
        assert graph_response.status_code == 200
        graph_payload = graph_response.json()
        assert graph_payload["stats"]["chapter_count"] == 2
        assert len(graph_payload["chapter_timeline"]) == 2


def test_upload_endpoint_accepts_pdf_with_text_layer(monkeypatch):
    class FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakeReader:
        def __init__(self, _stream):
            self.pages = [
                FakePage("Chapter 1\nLin enters the square."),
                FakePage("Chapter 2\nAya responds."),
            ]

    monkeypatch.setattr(ingest_parser, "PdfReader", FakeReader)
    with TestClient(app) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("api_pdf_upload.pdf", b"%PDF-test", "application/pdf")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["chapter_count"] == 2
        book_response = client.get(f"/api/books/{payload['book_id']}")
        assert book_response.status_code == 200
        assert book_response.json()["chapter_count"] == 2


def test_qa_and_summary_endpoints_use_persona_generation(monkeypatch):
    def fake_answer(*_args, **_kwargs):
        from backend.models import QuestionResponse

        return QuestionResponse(
            answer="鲁迅会先从眼前这段人物动作讲起。",
            persona_id="lu-xun",
            safe=True,
            reason="within_visible_scope",
            contexts=[],
            model_name="persona-qa-model",
        )

    def fake_summary(*_args, **_kwargs):
        from backend.models import SummaryResponse

        return SummaryResponse(
            summary="张爱玲式总结会先看关系里的冷暖与体面。",
            chapter_id="chapter-001",
            persona_id="zhang-ailing",
            model_name="persona-summary-model",
        )

    monkeypatch.setattr(app_module, "build_answer", fake_answer)
    monkeypatch.setattr(app_module, "summarize_chapter", fake_summary)

    with TestClient(app) as client:
        qa_response = client.post(
            "/api/qa",
            json={
                "book_id": "muse-demo-book",
                "question": "这段在说什么？",
                "highlight_text": "Lin opened the old notebook.",
                "current_chapter": 1,
                "persona_id": "lu-xun",
            },
        )
        assert qa_response.status_code == 200
        assert qa_response.json()["model_name"] == "persona-qa-model"

        summary_response = client.post(
            "/api/summary",
            json={
                "book_id": "muse-demo-book",
                "current_chapter": 1,
                "persona_id": "zhang-ailing",
            },
        )
        assert summary_response.status_code == 200
        assert summary_response.json()["model_name"] == "persona-summary-model"
