# Persona Agent Configuration

Muse Reading now ships with a complete runtime path for literary lead-reader agents. The three named agents are:

- `lu-xun`
- `mark-twain`
- `zhang-ailing`

There is also a `neutral` reader for non-persona output.

## What Is Wired End To End

Each persona agent now has all four layers required for direct use:

1. `Agent config`
   Defined in [services/persona/persona_service.py](/C:/Users/21358/Desktop/MuseReading/services/persona/persona_service.py), including display name, env var names, and prompt traits.
2. `Persona RAG knowledge base`
   Loaded from `data/processed/personas/persona_kb/<persona_id>/`.
3. `Prompt assembly`
   The system prompt, persona evidence, and reader-visible book context are assembled together before generation.
4. `Real model invocation`
   [services/persona/model_client.py](/C:/Users/21358/Desktop/MuseReading/services/persona/model_client.py) calls an OpenAI-compatible `/v1/chat/completions` endpoint.

## Required Environment Variables

Copy [`.env.example`](/C:/Users/21358/Desktop/MuseReading/.env.example) to `.env` and fill in your own local values. The backend loads this root-level `.env` file automatically on startup.

Expected variables:

- `LU_XUN_API_KEY`
- `LU_XUN_BASE_URL`
- `LU_XUN_MODEL_NAME`
- `MARK_TWAIN_API_KEY`
- `MARK_TWAIN_BASE_URL`
- `MARK_TWAIN_MODEL_NAME`
- `ZHANG_AILING_API_KEY`
- `ZHANG_AILING_BASE_URL`
- `ZHANG_AILING_MODEL_NAME`

Optional neutral reader variables:

- `MUSE_NEUTRAL_API_KEY`
- `MUSE_NEUTRAL_BASE_URL`
- `MUSE_NEUTRAL_MODEL_NAME`

If any required value is missing, the API returns a clear configuration error instead of a fake fallback answer.

## Runtime Behavior

For `POST /api/qa`:

1. The system retrieves only text visible up to `current_chapter`.
2. It queries the persona KB with the question, highlight, and visible context.
3. It builds a persona-specific system prompt.
4. It sends the persona prompt plus visible text evidence to the configured model endpoint.
5. It returns the generated answer together with `model_name`.

For `POST /api/summary`:

1. The system gathers the current chapter's visible chunks.
2. It retrieves supporting persona evidence.
3. It asks the configured persona model to summarize only that chapter.
4. It returns the generated summary together with `model_name`.

## Current Interface Surface

- `GET /api/personas`
- `GET /api/persona-agents`
- `GET /api/persona-agents/{persona_id}`
- `GET /api/persona-agents/{persona_id}/kb`
- `POST /api/persona-agents/{persona_id}/retrieve`
- `POST /api/persona-agents/{persona_id}/prompt-preview`
- `POST /api/qa`
- `POST /api/summary`

## Notes

- The persona model endpoint must be OpenAI-compatible.
- Persona style can shape interpretation and tone, but it never overrides spoiler boundaries.
- The persona KB stores structured evidence and summaries, not full copyrighted corpora.
