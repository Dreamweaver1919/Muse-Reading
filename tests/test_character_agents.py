from __future__ import annotations

from fastapi.testclient import TestClient

import api.app as app_module
from backend.models import (
    CharacterCandidate,
    CharacterChatResponse,
    CharacterProfile,
    CharacterRelationship,
    InlineBubble,
)
from services.ingest.parser import build_book_record
from services.character.service import _heuristic_character_candidates


app = app_module.app


def demo_book():
    text = """
Chapter 1
Ursula looked at Jose Arcadio and spoke softly.

Aureliano kept silent while Ursula watched him.

Chapter 2
Remedios laughed, and Ursula turned back toward Aureliano.
""".strip()
    return build_book_record("demo_characters", text, __file__)


def test_heuristic_character_candidates_collect_repeated_names():
    book = demo_book()
    candidates = _heuristic_character_candidates(book.chunks, current_chapter=2, limit=6)
    names = {candidate.character_name for candidate in candidates}
    assert "Ursula" in names
    assert "Aureliano" in names


def test_heuristic_character_candidates_filter_discourse_words_from_chinese_text():
    text = """
第1章
何塞站在门口。实际上，他说自己已经想清楚了。

乌尔苏拉看着何塞，她说这件事还没有结束。

何塞又一次提到布恩迪亚家族的命运。
""".strip()
    book = build_book_record("demo_chinese_characters", text, __file__)
    candidates = _heuristic_character_candidates(book.chunks, current_chapter=1, limit=8)
    names = {candidate.character_name for candidate in candidates}
    assert "何塞" in names
    assert "实际上" not in names
    assert "他说" not in names
    assert "她说" not in names


def test_character_endpoints_are_available_with_monkeypatched_services(monkeypatch):
    def fake_candidates(book, current_chapter: int, limit: int = 10):
        return [
            CharacterCandidate(
                character_id="char-ursula",
                character_name="Ursula",
                mention_count=6,
                chapter_hits=[1, 2],
                preview="她总在观察家人的变化。",
            )
        ]

    def fake_profile(book, character_name: str, current_chapter: int):
        return CharacterProfile(
            character_id="char-ursula",
            character_name=character_name,
            summary="她是家庭秩序与情感压力的中心人物。",
            core_traits=["清醒", "承担", "敏锐"],
            relationships=[CharacterRelationship(target="Aureliano", description="她始终留意他的沉默。")],
            signature_tension="她不断在维持秩序与看见裂缝之间摆动。",
            evidence_chunk_ids=["demo_characters-c001-p001"],
            current_scope="目前已读范围只足以看出她在家庭中的中心位置。",
            model_name="test-model",
        )

    def fake_chat(book, character_name: str, question: str, current_chapter: int, conversation_history=None, top_k: int = 6):
        profile = fake_profile(book, character_name, current_chapter)
        return CharacterChatResponse(
            answer="如果从乌尔苏拉的视角看，她最先注意到的是沉默背后的变化。",
            character_name=character_name,
            safe=True,
            reason="within_visible_scope",
            model_name="test-model",
            profile=profile,
        )

    def fake_bubbles(book, current_chapter: int, visible_chunk_ids, persona_id: str, assistant_mode: str, character_name: str, max_bubbles: int):
        return [
            InlineBubble(
                bubble_id="bubble-1",
                chunk_id=visible_chunk_ids[0],
                anchor_text="spoke softly",
                label="关系",
                comment="语气的放低说明她在试探对方反应。",
                emphasis="relation",
            )
        ]

    monkeypatch.setattr(app_module, "list_character_candidates", fake_candidates)
    monkeypatch.setattr(app_module, "generate_character_profile", fake_profile)
    monkeypatch.setattr(app_module, "answer_as_character", fake_chat)
    monkeypatch.setattr(app_module, "generate_inline_bubbles", fake_bubbles)

    with TestClient(app) as client:
        books = client.get("/api/books").json()
        book_id = books[0]["book_id"]

        candidates = client.get(f"/api/books/{book_id}/characters?current_chapter=1")
        assert candidates.status_code == 200
        assert candidates.json()[0]["character_name"] == "Ursula"

        profile = client.post(
            f"/api/books/{book_id}/characters/profile",
            json={"book_id": book_id, "character_name": "Ursula", "current_chapter": 1},
        )
        assert profile.status_code == 200
        assert profile.json()["core_traits"] == ["清醒", "承担", "敏锐"]

        chat = client.post(
            f"/api/books/{book_id}/characters/chat",
            json={
                "book_id": book_id,
                "character_name": "Ursula",
                "question": "她现在最在意什么？",
                "current_chapter": 1,
                "conversation_history": [{"role": "user", "content": "先别剧透。"}],
            },
        )
        assert chat.status_code == 200
        assert "乌尔苏拉" in chat.json()["answer"]

        bubbles = client.post(
            f"/api/books/{book_id}/inline-bubbles",
            json={
                "book_id": book_id,
                "current_chapter": 1,
                "visible_chunk_ids": ["demo_characters-c001-p001"],
                "persona_id": "persona_lu_xun",
                "assistant_mode": "persona",
                "character_name": "",
                "max_bubbles": 3,
            },
        )
        assert bubbles.status_code == 200
        assert bubbles.json()[0]["anchor_text"] == "spoke softly"
