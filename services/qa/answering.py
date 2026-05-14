from __future__ import annotations

from backend.models import QuestionRequest, QuestionResponse, RetrievedContext
from services.graph.storage import load_graph
from services.orchestration.models import ReadingProgress, SelectionAnchor, SelectionContext
from services.orchestration.service import OrchestrationService
from services.persona.persona_service import generate_persona_response
from services.qa.retrieval import retrieve_chunks
from services.safety.anti_spoiler import is_spoiler_question


def _merge_contexts(local_hits: list[RetrievedContext], graph_hits) -> list[RetrievedContext]:
    merged: dict[str, RetrievedContext] = {hit.chunk_id: hit for hit in local_hits}
    for hit in graph_hits:
        paragraph_index = hit.paragraph_id if hit.paragraph_id is not None else 0
        merged.setdefault(
            hit.chunk_id,
            RetrievedContext(
                chunk_id=hit.chunk_id,
                chapter_index=hit.chapter_id,
                paragraph_index=paragraph_index,
                score=1.0,
                text=hit.text,
            ),
        )
    ranked = sorted(merged.values(), key=lambda item: (item.score, -item.chapter_index, -item.paragraph_index), reverse=True)
    return ranked


def build_answer(request: QuestionRequest, chunks) -> QuestionResponse:
    safety = is_spoiler_question(request.question)
    try:
        graph = load_graph(request.book_id)
    except FileNotFoundError:
        graph = None

    orchestration = OrchestrationService().orchestrate(
        chunks=chunks,
        request_id=f"qa-{request.book_id}-{request.current_chapter}",
        book_id=request.book_id,
        query=request.question,
        reading_progress=ReadingProgress(
            book_id=request.book_id,
            chapter_id=request.current_chapter,
            paragraph_id=9999,
            token_offset=10**9,
        ),
        selection_context=SelectionContext(
            book_id=request.book_id,
            selected_text=request.highlight_text,
            anchor=SelectionAnchor(chapter_id=request.current_chapter, paragraph_id=0),
        ),
        top_k=request.top_k,
        temporal_graph=graph,
    )
    local_contexts = retrieve_chunks(
        chunks=chunks,
        query=f"{request.highlight_text} {request.question}".strip(),
        max_chapter=request.current_chapter,
        top_k=request.top_k,
    )
    contexts = _merge_contexts(local_contexts, orchestration.hits)[: request.top_k]
    visible_context_texts = [context.text for context in contexts]

    if not safety.safe:
        refusal, model_name, _ = generate_persona_response(
            persona_id=request.persona_id,
            task="qa",
            book_title=request.book_id,
            question=(
                "用户的问题超出了已读范围，请拒绝剧透，并把话题收回当前已读内容。"
                f"\n原问题：{request.question}"
            ),
            visible_contexts=visible_context_texts,
            current_chapter=request.current_chapter,
            highlight_text=request.highlight_text,
            top_k=request.top_k,
            conversation_history=request.conversation_history,
        )
        return QuestionResponse(
            answer=refusal,
            persona_id=request.persona_id,
            safe=False,
            reason=safety.reason,
            contexts=contexts,
            model_name=model_name,
        )

    if not visible_context_texts:
        answer, model_name, _ = generate_persona_response(
            persona_id=request.persona_id,
            task="qa",
            book_title=request.book_id,
            question=(
                "当前没有检索到足够正文证据。请用中文明确说明证据不足，"
                "并引导用户改问更贴近当前段落的问题。"
            ),
            visible_contexts=[],
            current_chapter=request.current_chapter,
            highlight_text=request.highlight_text,
            top_k=request.top_k,
            conversation_history=request.conversation_history,
        )
        return QuestionResponse(
            answer=answer,
            persona_id=request.persona_id,
            safe=True,
            reason="no_visible_context",
            contexts=[],
            model_name=model_name,
        )

    answer, model_name, _ = generate_persona_response(
        persona_id=request.persona_id,
        task="qa",
        book_title=request.book_id,
        question=request.question,
        visible_contexts=visible_context_texts,
        current_chapter=request.current_chapter,
        highlight_text=request.highlight_text,
        top_k=request.top_k,
        conversation_history=request.conversation_history,
    )
    return QuestionResponse(
        answer=answer,
        persona_id=request.persona_id,
        safe=True,
        reason=safety.reason,
        contexts=contexts,
        model_name=model_name,
    )
