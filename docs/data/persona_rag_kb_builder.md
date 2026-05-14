# Persona RAG KB Builder

This document describes the local builder that converts the three lead-reader persona catalogs and persona packs into a retrieval-ready knowledge-base skeleton for traditional RAG.

## Goal

The builder prepares persona assets for prompt injection and lightweight retrieval without putting persona nodes into the temporal graph database.

Current supported personas:

- `persona_lu_xun`
- `persona_mark_twain`
- `persona_zhang_ailing`

## Inputs

The builder reads two existing asset types:

- `data/raw/persona_sources/catalog_<persona>__v001.json`
  - source inventory split into:
    - `works`
    - `voice_sources`
    - `biography_and_critical`
- `data/processed/personas/persona_<name>__v*.json`
  - schema-conformant persona pack with:
    - `fact_layer`
    - `style_layer`
    - `stance_layer`
    - `source_layer`
    - `constraints`

## Outputs

For each persona, the builder writes a directory under `data/processed/personas/persona_kb/<persona_id>/`:

- `documents.jsonl`
  - one `persona_profile` document
  - one `source_document` per catalog entry
- `retrieval_snippets.jsonl`
  - persona-pack snippets for style, stance, themes, and boundary
  - one `source_overview` snippet per catalog entry
- `manifest.json`
  - input references
  - output file references
  - category counts
  - retrieval notes

## Retrieval usage

Recommended retrieval order:

1. Use the current reader-visible book context from the book-side RAG or temporal graph.
2. Retrieve persona snippets from `persona_pack` when the task needs voice, stance, or style control.
3. Retrieve `voice_sources` first when the task is "how would this persona comment on the passage?"
4. Use `works` and `biography_and_critical` as supporting evidence for background and recurring motifs.

## Run

```bash
python scripts/persona_rag_kb_builder.py
```

Build a single persona:

```bash
python scripts/persona_rag_kb_builder.py --persona-id persona_lu_xun
```

## Notes

- The builder does not use or store API keys.
- The KB is intentionally summary-based; it does not attempt to ship full copyrighted text.
- `voice_sources` is the highest-value bucket for lead-reader prompting because it captures tone, reasoning rhythm, and self-positioning.
