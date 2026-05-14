# Temporal Context Graph

Muse Reading currently treats the temporal context graph as a local graph store layered on top of the chunked book record rather than as a separate external database.

## What The Graph Stores

- `chapter` nodes: chapter-level timeline anchors and browse summaries
- `episode` nodes: paragraph or chunk level narrative units with provenance
- `entity` nodes: characters, locations, groups, concepts, and themes
- `relation` edges: co-presence, dialogue, and conflict relations with chapter validity
- `community` nodes: connected components over entity adjacency
- `saga` nodes: contiguous multi-chapter narrative threads

## Timeline Layer

Each graph includes a `chapter_timeline` array. Every entry records:

- `chapter_index`
- `episode_ids`
- `entity_ids`
- `relation_ids`
- `community_ids`
- `saga_ids`
- `spoiler_level`
- `summary`

This gives API consumers a stable chapter-by-chapter browse surface without needing to reconstruct timeline state from raw edges.

## Query Layer

`services/graph/retrieval.py` supports progress-aware retrieval with filters for:

- `max_chapter` and `min_chapter`
- `entity_names`
- `entity_types`
- `relation_types`
- `node_types`
- `metadata_filters`
- `min_entity_mentions`
- `min_relation_weight`

The retrieval result also returns `hit_type_breakdown`, `applied_filters`, and graph-level stats so higher layers can inspect what the graph actually surfaced.

## Storage Layer

Graphs are persisted as JSON under `workspace_state/graphs/`. The storage metadata currently records:

- `storage_version`
- `saved_at`
- `graph_path`

This is still a local file-backed graph store. Future work can swap the storage backend while preserving the same graph model and retrieval surface.

## API Surface

The graph layer is now exposed through the FastAPI app:

- `GET /api/books/{book_id}/graph`
  - Returns graph stats plus browsable `chapters`, `chapter_timeline`, `episodes`, `entities`, `relations`, `communities`, and `sagas`
- `GET /api/books/{book_id}/graph/metadata`
  - Returns storage metadata and graph stats only
- `POST /api/books/{book_id}/graph/query`
  - Runs progress-aware graph retrieval using the `GraphQuery` schema

This keeps the graph store local and file-backed while still giving the frontend and evaluation scripts a stable database-style query surface.
