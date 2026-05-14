from __future__ import annotations

from services.persona.persona_service import (
    build_persona_prompt_preview,
    get_persona_agent,
    get_persona_kb_manifest,
    list_persona_agents,
    resolve_persona_runtime,
    retrieve_persona_snippets,
)
from backend.models import PersonaPromptPreviewRequest, PersonaRAGQueryRequest


def test_persona_agents_are_exposed_with_catalog_counts():
    agents = list_persona_agents()
    agent_ids = {agent.agent_id for agent in agents}
    assert {"neutral", "lu-xun", "mark-twain", "zhang-ailing"}.issubset(agent_ids)

    lu_xun = get_persona_agent("persona_lu_xun")
    assert lu_xun.catalog_summary.total_sources >= 20
    assert lu_xun.catalog_summary.voice_sources >= 10


def test_persona_kb_manifest_is_available():
    manifest = get_persona_kb_manifest("mark-twain")
    assert manifest["persona_id"] == "persona_mark_twain"
    assert manifest["document_counts"]["works"] >= 10


def test_persona_snippet_retrieval_returns_ranked_hits():
    hits = retrieve_persona_snippets(
        "zhang-ailing",
        PersonaRAGQueryRequest(query="urban desire family power atmosphere", top_k=3),
    )
    assert len(hits) >= 1
    assert hits[0].score > 0


def test_persona_prompt_preview_contains_context():
    preview = build_persona_prompt_preview(
        "lu-xun",
        PersonaPromptPreviewRequest(
            book_context="A passage about obedience, habit, and quiet pressure in daily life.",
            question="How would this persona comment on social numbness?",
            top_k=3,
        ),
    )
    assert preview.persona_id == "persona_lu_xun"
    assert "你是 鲁迅 风格的中文阅读陪伴 agent" in preview.system_prompt
    assert preview.retrieved_hits
    assert "persona_pack" in preview.persona_context or "voice_sources" in preview.persona_context


def test_resolve_persona_runtime_requires_complete_env(monkeypatch):
    monkeypatch.delenv("LU_XUN_API_KEY", raising=False)
    monkeypatch.delenv("LU_XUN_BASE_URL", raising=False)
    monkeypatch.delenv("LU_XUN_MODEL_NAME", raising=False)
    try:
        resolve_persona_runtime("lu-xun")
    except RuntimeError as exc:
        assert "LU_XUN_API_KEY" in str(exc)
        assert "LU_XUN_BASE_URL" in str(exc)
        assert "LU_XUN_MODEL_NAME" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("resolve_persona_runtime should fail when env vars are missing")
