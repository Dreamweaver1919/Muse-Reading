from __future__ import annotations

from backend.models import SummaryResponse
from services.persona.persona_service import generate_persona_response


def summarize_chapter(book, current_chapter: int, persona_id: str) -> SummaryResponse:
    chapter_chunks = [chunk for chunk in book.chunks if chunk.chapter_index == current_chapter]
    visible_contexts = [chunk.text.strip() for chunk in chapter_chunks if chunk.text.strip()]
    summary, model_name, _ = generate_persona_response(
        persona_id=persona_id,
        task="summary",
        book_title=book.title,
        question=f"请总结这本书第 {current_chapter} 章目前已读范围的内容。",
        visible_contexts=visible_contexts[:8],
        current_chapter=current_chapter,
        top_k=5,
    )
    return SummaryResponse(
        summary=summary,
        chapter_id=f"chapter-{current_chapter:03d}",
        persona_id=persona_id,
        model_name=model_name,
    )
