# Muse Reading

Muse Reading is an AI reading workspace for long-form text. It combines uploadable book text, progress-aware retrieval, inline highlight QA, chapter summaries, lightweight persona-guided companionship, and anti-spoiler controls into a single MVP that can be run locally and extended as an open-source project.

This repository is the current engineering scaffold for that idea. It is built around one principle: get the reading loop working first, then deepen the intelligence with better datasets, richer temporal graphs, stronger persona grounding, and more complete evaluation.

## Project Goals

- Turn uploaded reading text into a structured, retrievable reading corpus.
- Support immersive reading with paragraph-level navigation and reading progress tracking.
- Answer user questions from highlighted text without leaking future plot.
- Generate chapter-level summaries based on current reading stage.
- Support persona-guided companion reading as a pluggable layer.
- Build a dataset and evaluation pipeline around retrieval, anti-spoiler behavior, and reading understanding.

## Core Features

- Text upload and local book ingestion for `.txt`, `.pdf`, and `.epub` files.
- Chapter and paragraph parsing with chunk-level metadata.
- Temporal graph generation from uploaded book content.
- Reader UI with chapter navigation, paragraph selection, and payload preview.
- Highlight-triggered QA with progress-aware retrieval.
- Chapter summary endpoint with persona-aware generation.
- Complete runtime path for Lu Xun, Mark Twain, and Zhang Ailing lead-reader agents through persona RAG plus OpenAI-compatible model endpoints.
- Benchmark fixtures for `highlight_qa`, `anti_spoiler`, and `chapter_summary`.

## System Architecture

Muse Reading currently has four cooperating layers:

1. `Frontend interaction layer`
   The static web reader in [frontend/public](/C:/Users/21358/Desktop/MuseReading/frontend/public) handles upload, chapter navigation, paragraph selection, summary triggers, and question submission.

2. `Application and orchestration layer`
   The FastAPI app in [api/app.py](/C:/Users/21358/Desktop/MuseReading/api/app.py) exposes upload, book, persona, QA, orchestration, summary, and graph endpoints.

3. `Knowledge and retrieval layer`
   The backend builds normalized book records, retrieval chunks, and a temporal context graph from uploaded text. Retrieval is progress-aware and designed to support anti-spoiler filtering before answer generation.

4. `Dataset and evaluation layer`
   The repository includes schemas, manifests, examples, benchmark fixtures, and evaluation scripts so that ingestion, QA, summary, and anti-spoiler behavior can be regression-tested and expanded into a fuller benchmark suite.

### Current Runtime Flow

```text
upload text
  -> normalize into book record
  -> parse chapters and paragraphs
  -> build retrieval chunks
  -> build temporal graph
  -> read in frontend
  -> ask highlight question or request chapter summary
  -> retrieve only visible context
  -> generate answer with persona style and spoiler guard
```

## Dataset Construction Strategy

The repository follows a schema-first, metadata-first structure rather than bundling large copyrighted corpora.

### Data Categories

- `book text corpus`
  User-uploaded text, demo text, and future public-domain or licensed books.
- `persona source corpus`
  Materials used to shape lead-reader agents, such as essays, letters, speeches, prefaces, biographies, and criticism.
- `annotation data`
  `highlight_qa`, `chapter_evolution`, salience labels, and future reading-session annotations.
- `evaluation data`
  Retrieval, persona consistency, anti-spoiler, and user-study-oriented evaluation packages.

### Current Data Layout

```text
data/
  raw/
    books/
    persona_sources/
  processed/
    books/
    personas/
  annotations/
    highlight_qa/
    chapter_evolution/
  eval/
    retrieval/
    persona_consistency/
    anti_spoiler/
  manifests/
```

### Layered Text Representation

The current project reserves a hierarchical representation for uploaded book text:

- `L0`: raw paragraph units
- `L1`: retrieval-ready chunks
- `L2`: chapter structure summaries
- `L3`: global topic or route index
- `L4`: quote, stance, or commentary-ready layer

The first practical use of this hierarchy already exists in the local dataset-building scripts and graph export pipeline.

## Quick Start

### Requirements

- Python `3.10+` recommended
- `pip`

### Install

```bash
python -m pip install -r requirements.txt
```

### Configure Persona Agents

To use `lu-xun`, `mark-twain`, or `zhang-ailing`, copy [`.env.example`](/C:/Users/21358/Desktop/MuseReading/.env.example) to `.env` and fill in your own OpenAI-compatible endpoint, model name, and API key for each agent. The app reads this root-level `.env` automatically at startup.

### Run The API And Reader

```bash
uvicorn api.app:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

On startup, the app auto-loads the bundled demo book from [examples/muse_demo_book.txt](/C:/Users/21358/Desktop/MuseReading/examples/muse_demo_book.txt) if it exists.

## Frontend Usage

The reader UI is served directly by FastAPI from [frontend/public/index.html](/C:/Users/21358/Desktop/MuseReading/frontend/public/index.html).

Current UI capabilities:

- upload a `.txt`, `.pdf`, or `.epub` book
- browse chapter and paragraph content
- select a paragraph as reading focus
- inspect `reading_progress` and `selection_context`
- ask a question from the current reading context
- request a chapter summary
- switch personas from the local persona registry

Screenshot placeholder:

- add future screenshots under `docs/` or a dedicated `screenshots/` folder before public release

## API Overview

Current endpoints exposed by [api/app.py](/C:/Users/21358/Desktop/MuseReading/api/app.py):

- `GET /api/health`
- `GET /api/books`
- `GET /api/books/{book_id}`
- `GET /api/books/{book_id}/graph`
- `GET /api/personas`
- `POST /api/upload`
- `POST /api/qa`
- `POST /api/orchestrate`
- `POST /api/summary`

### Minimal API Notes

- `POST /api/upload`
  Uploads a `.txt`, `.pdf`, or `.epub` file, parses it into a book record, and builds a temporal graph.
- `POST /api/qa`
  Accepts `book_id`, `question`, optional `highlight_text`, `current_chapter`, and `persona_id`.
- `POST /api/orchestrate`
  Runs mixed retrieval over visible chunks and graph context.
- `POST /api/summary`
  Generates a current-chapter summary with an optional persona style.

## Repository Structure

```text
api/                 FastAPI entrypoint
architecture/        interface and system design notes
backend/             core models, config, storage
benchmarks/          benchmark fixtures for smoke evaluation
data/                raw, processed, annotation, eval, and manifest assets
docs/                architecture and data design documentation
eval/                evaluation runners
examples/            demo reading text
frontend/            static reader UI
schemas/             JSON schema definitions
scripts/             dataset and registry builders
services/            ingestion, graph, orchestration, qa, persona, safety, summary
tests/               regression tests
workspace_state/     local runtime artifacts such as saved books and graphs
```

## Evaluation

The current repository includes a minimal but working evaluation scaffold.

### Run Benchmarks

```bash
python eval/run_eval.py
```

### Run Tests

```bash
pytest -q
```

### What Is Covered Today

- `highlight_qa`
  Checks whether expected support chunks are retrieved and answers are returned.
- `anti_spoiler`
  Checks whether future-plot questions are refused or constrained correctly.
- `chapter_summary`
  Checks whether summaries include expected phrases and avoid forbidden ones.

These are smoke and regression checks, not full leaderboard-grade benchmarks yet.

## Current Limitations

- `pdf` support currently expects a text-layer PDF rather than a scanned image PDF.
- Temporal graph extraction is heuristic and lightweight.
- Frontend copy contains placeholder or draft text in several places.
- Persona output depends on locally configured model credentials in `.env`.
- Evaluation is still small and synthetic compared with the intended benchmark scope.
- Copyright-sensitive corpora are represented mostly through manifests and examples rather than full released text.

## Roadmap

- Add richer ingestion for `epub`, `docx`, and controlled `pdf` workflows.
- Strengthen temporal graph extraction and graph-aware retrieval.
- Expand Chinese lead-reader personas and persona consistency evaluation.
- Build larger retrieval, narrative understanding, long-dialog, and anti-spoiler benchmarks.
- Add screenshot assets, deployment instructions, and packaging for public release.
- Improve frontend text quality and polish the reading interaction loop.

## Open-Source Release Notes

This repository is structured to be safely open-sourced:

- schemas, manifests, examples, and scripts are included
- benchmark fixtures are tiny and synthetic
- copyrighted book content should remain out of public releases unless redistribution rights are explicit
- public-domain or licensed content can be added later through the existing data structure

## Related Project Docs

- [Architecture alignment](/C:/Users/21358/Desktop/MuseReading/docs/architecture_alignment.md)
- [README architecture summary](/C:/Users/21358/Desktop/MuseReading/docs/readme_architecture_summary.md)
- [Data design](/C:/Users/21358/Desktop/MuseReading/docs/data/muse_reading_data_design.md)
- [Benchmarks README](/C:/Users/21358/Desktop/MuseReading/benchmarks/README.md)
- [Data skeleton README](/C:/Users/21358/Desktop/MuseReading/data/README.md)
