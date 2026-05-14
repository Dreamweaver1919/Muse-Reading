from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import ROOT_DIR, UPLOADS_DIR
from backend.models import (
    CharacterChatRequest,
    CharacterProfileRequest,
    InlineBubbleRequest,
    PersonaPromptPreviewRequest,
    PersonaRAGQueryRequest,
    QuestionRequest,
    SummaryRequest,
    UploadResponse,
)
from backend.storage import list_books, load_book, save_book
from services.graph.builder import build_temporal_graph
from services.graph.models import GraphQuery
from services.graph.retrieval import TemporalGraphRetriever
from services.graph.storage import load_graph, load_graph_metadata, save_graph
from services.ingest.parser import (
    SUPPORTED_UPLOAD_SUFFIXES,
    UploadTextExtractionError,
    UnsupportedUploadFormatError,
    build_book_record,
    build_book_record_from_upload,
    read_uploaded_text,
    slugify,
)
from services.orchestration.service import OrchestrationService
from services.character.service import (
    answer_as_character,
    generate_character_profile,
    generate_inline_bubbles,
    list_character_candidates,
)
from services.persona.persona_service import (
    build_persona_prompt_preview,
    get_persona_agent,
    get_persona_kb_manifest,
    list_persona_agents,
    list_personas,
    PersonaAgentConfigurationError,
    PersonaAgentInvocationError,
    retrieve_persona_snippets,
)
from services.qa.answering import build_answer
from services.summary.chapter_summary import summarize_chapter


app = FastAPI(title="Muse Reading MVP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = ROOT_DIR / "frontend" / "public"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


def ensure_demo_book_loaded() -> None:
    demo_path = ROOT_DIR / "examples" / "muse_demo_book.txt"
    if not demo_path.exists():
        return
    title = demo_path.stem
    record = build_book_record(title=title, raw_text=demo_path.read_text(encoding="utf-8"), source_path=demo_path)
    save_book(record)
    save_graph(build_temporal_graph(record))


def get_or_build_book(book_id: str):
    try:
        record = load_book(book_id)
    except FileNotFoundError:
        demo_path = ROOT_DIR / "examples" / f"{book_id}.txt"
        if not demo_path.exists():
            raise
        record = build_book_record(
            title=demo_path.stem,
            raw_text=demo_path.read_text(encoding="utf-8"),
            source_path=demo_path,
        )
        save_book(record)
    if record.chunks:
        return record
    source_path = Path(record.source_path)
    if not source_path.exists():
        return record
    rebuilt = build_book_record_from_upload(
        title=record.title,
        filename=source_path.name,
        raw_bytes=source_path.read_bytes(),
        source_path=source_path,
    )
    if rebuilt.chunks:
        save_book(rebuilt)
        save_graph(build_temporal_graph(rebuilt))
        return rebuilt
    return record


def get_or_build_graph(book_id: str):
    try:
        return load_graph(book_id)
    except FileNotFoundError:
        book = get_or_build_book(book_id)
        graph = build_temporal_graph(book)
        save_graph(graph)
        return graph


@app.on_event("startup")
def startup_event() -> None:
    ensure_demo_book_loaded()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/books")
def books() -> list[dict[str, str]]:
    return list_books()


@app.get("/api/personas")
def personas():
    return [persona.model_dump() for persona in list_personas()]


@app.get("/api/persona-agents")
def persona_agents():
    return [agent.model_dump() for agent in list_persona_agents()]


@app.get("/api/persona-agents/{persona_id}")
def persona_agent_detail(persona_id: str):
    return get_persona_agent(persona_id).model_dump()


@app.get("/api/persona-agents/{persona_id}/kb")
def persona_agent_kb(persona_id: str):
    return get_persona_kb_manifest(persona_id)


@app.post("/api/persona-agents/{persona_id}/retrieve")
def persona_agent_retrieve(persona_id: str, request: PersonaRAGQueryRequest):
    return [hit.model_dump() for hit in retrieve_persona_snippets(persona_id, request)]


@app.post("/api/persona-agents/{persona_id}/prompt-preview")
def persona_agent_prompt_preview(persona_id: str, request: PersonaPromptPreviewRequest):
    return build_persona_prompt_preview(persona_id, request).model_dump()


@app.get("/api/books/{book_id}")
def book_detail(book_id: str):
    book = get_or_build_book(book_id)
    chapters: dict[int, list[dict[str, str | int]]] = {}
    for chunk in book.chunks:
        chapters.setdefault(chunk.chapter_index, []).append(
            {
                "chunk_id": chunk.chunk_id,
                "paragraph_index": chunk.paragraph_index,
                "text": chunk.text,
            }
        )
    return {
        "book_id": book.book_id,
        "title": book.title,
        "chapter_count": book.chapter_count,
        "chapters": chapters,
    }


@app.get("/api/books/{book_id}/characters")
def book_characters(book_id: str, current_chapter: int = 1, limit: int = 10):
    try:
        book = get_or_build_book(book_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="book_not_found") from exc
    try:
        return [item.model_dump() for item in list_character_candidates(book, current_chapter, limit=limit)]
    except (PersonaAgentConfigurationError, PersonaAgentInvocationError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/books/{book_id}/characters/profile")
def character_profile(book_id: str, request: CharacterProfileRequest):
    try:
        book = get_or_build_book(book_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="book_not_found") from exc
    try:
        return generate_character_profile(book, request.character_name, request.current_chapter).model_dump()
    except (PersonaAgentConfigurationError, PersonaAgentInvocationError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/books/{book_id}/characters/chat")
def character_chat(book_id: str, request: CharacterChatRequest):
    try:
        book = get_or_build_book(book_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="book_not_found") from exc
    try:
        return answer_as_character(
            book,
            character_name=request.character_name,
            question=request.question,
            current_chapter=request.current_chapter,
            conversation_history=request.conversation_history,
            top_k=request.top_k,
        ).model_dump()
    except (PersonaAgentConfigurationError, PersonaAgentInvocationError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/books/{book_id}/inline-bubbles")
def inline_bubbles(book_id: str, request: InlineBubbleRequest):
    try:
        book = get_or_build_book(book_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="book_not_found") from exc
    try:
        return [
            item.model_dump()
            for item in generate_inline_bubbles(
                book,
                current_chapter=request.current_chapter,
                visible_chunk_ids=request.visible_chunk_ids,
                persona_id=request.persona_id,
                assistant_mode=request.assistant_mode,
                character_name=request.character_name,
                max_bubbles=request.max_bubbles,
            )
        ]
    except (PersonaAgentConfigurationError, PersonaAgentInvocationError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/books/{book_id}/graph")
def graph_detail(book_id: str):
    graph = get_or_build_graph(book_id)
    return {
        "graph_id": graph.graph_id,
        "book_id": graph.book_id,
        "title": graph.title,
        "graph_version": graph.graph_version,
        "stats": graph.stats().model_dump(),
        "metadata": graph.metadata,
        "chapters": [chapter.model_dump() for chapter in graph.chapters.head(10)],
        "chapter_timeline": [item.model_dump() for item in graph.chapter_timeline[:10]],
        "episodes": [episode.model_dump() for episode in graph.episodes.head(10)],
        "entities": [entity.model_dump() for entity in graph.entities.head(20)],
        "relations": [relation.model_dump() for relation in graph.relations.head(20)],
        "communities": [community.model_dump() for community in graph.communities.head(10)],
        "sagas": [saga.model_dump() for saga in graph.sagas.head(10)],
    }


@app.get("/api/books/{book_id}/graph/metadata")
def graph_metadata(book_id: str):
    try:
        return load_graph_metadata(book_id)
    except FileNotFoundError:
        graph = get_or_build_graph(book_id)
        return {
            "graph_id": graph.graph_id,
            "book_id": graph.book_id,
            "title": graph.title,
            "graph_version": graph.graph_version,
            "storage": graph.metadata.get("storage", {}),
            "stats": graph.stats().model_dump(),
        }


@app.post("/api/books/{book_id}/graph/query")
def graph_query(book_id: str, query: GraphQuery):
    graph = get_or_build_graph(book_id)
    effective_query = query
    if not effective_query.query and not effective_query.node_types:
        effective_query = effective_query.model_copy(update={"node_types": ["chapter", "episode"]})
    result = TemporalGraphRetriever().retrieve(graph, effective_query)
    return result.model_dump()


@app.post("/api/upload", response_model=UploadResponse)
async def upload_book(file: UploadFile = File(...)) -> UploadResponse:
    original_name = file.filename or "uploaded.txt"
    suffix = Path(original_name).suffix.lower() or ".txt"
    raw_bytes = await file.read()
    try:
        text = read_uploaded_text(original_name, raw_bytes)
    except UnsupportedUploadFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except UploadTextExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    title = Path(original_name).stem
    safe_name = slugify(title)
    upload_path = UPLOADS_DIR / f"{safe_name}{suffix}"
    if suffix in SUPPORTED_UPLOAD_SUFFIXES - {".txt"}:
        upload_path.write_bytes(raw_bytes)
    else:
        upload_path.write_text(text, encoding="utf-8")
    record = build_book_record_from_upload(
        title=title,
        filename=original_name,
        raw_bytes=raw_bytes if suffix != ".txt" else text.encode("utf-8"),
        source_path=upload_path,
    )
    save_book(record)
    save_graph(build_temporal_graph(record))
    return UploadResponse(
        book_id=record.book_id,
        title=record.title,
        chapter_count=record.chapter_count,
        chunk_count=len(record.chunks),
    )


@app.post("/api/qa")
def ask_question(request: QuestionRequest):
    try:
        book = get_or_build_book(request.book_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="book_not_found") from exc
    try:
        return build_answer(request, book.chunks)
    except PersonaAgentConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PersonaAgentInvocationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/orchestrate")
def orchestrate(payload: dict):
    book_id = payload["book_id"]
    book = load_book(book_id)
    graph = get_or_build_graph(book_id)
    service = OrchestrationService()
    result = service.orchestrate(
        chunks=book.chunks,
        request_id=payload.get("request_id", f"orchestrate-{book_id}"),
        book_id=book_id,
        query=payload.get("query", ""),
        reading_progress=payload["reading_progress"],
        selection_context=payload.get("selection_context"),
        top_k=payload.get("top_k", 6),
        temporal_graph=graph,
    )
    return result.model_dump()


@app.post("/api/summary")
def chapter_summary(request: SummaryRequest):
    try:
        book = get_or_build_book(request.book_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="book_not_found") from exc
    try:
        return summarize_chapter(book, request.current_chapter, request.persona_id)
    except PersonaAgentConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PersonaAgentInvocationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
