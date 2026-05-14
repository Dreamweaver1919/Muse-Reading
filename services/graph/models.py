from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, Literal

from pydantic import BaseModel, Field


NodeKind = Literal["chapter", "episode", "entity", "relation", "community", "saga"]
EntityType = Literal["character", "location", "artifact", "group", "theme", "concept", "unknown"]


class NodeTable(dict[str, Any]):
    """A dict-like store that also supports API-friendly slice access."""

    def __getitem__(self, key: object) -> Any:
        if isinstance(key, slice):
            return list(self.values())[key]
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)

    def head(self, limit: int) -> list[Any]:
        return list(self.values())[:limit]

    def ids(self) -> list[str]:
        return list(self.keys())

    def where(self, predicate: Callable[[Any], bool]) -> list[Any]:
        return [item for item in self.values() if predicate(item)]


class GraphProvenance(BaseModel):
    chunk_id: str
    book_id: str
    chapter_id: str
    chapter_index: int
    paragraph_id: str
    paragraph_index: int
    text_excerpt: str
    source: Literal["chunk", "chapter", "episode", "relation", "community", "saga"] = "chunk"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChapterTimelineEntry(BaseModel):
    chapter_id: str
    chapter_index: int
    title: str = ""
    episode_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    community_ids: list[str] = Field(default_factory=list)
    saga_ids: list[str] = Field(default_factory=list)
    spoiler_level: int = 0
    paragraph_count: int = 0
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[GraphProvenance] = Field(default_factory=list)


class ChapterNode(BaseModel):
    node_kind: Literal["chapter"] = "chapter"
    chapter_node_id: str
    book_id: str
    chapter_id: str
    chapter_index: int
    title: str = ""
    episode_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    community_ids: list[str] = Field(default_factory=list)
    saga_ids: list[str] = Field(default_factory=list)
    spoiler_level: int = 0
    paragraph_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[GraphProvenance] = Field(default_factory=list)


class EpisodeNode(BaseModel):
    node_kind: Literal["episode"] = "episode"
    episode_id: str
    book_id: str
    chunk_id: str
    chapter_id: str
    chapter_index: int
    paragraph_id: str
    paragraph_index: int
    text: str
    spoiler_level: int = 0
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    community_ids: list[str] = Field(default_factory=list)
    saga_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[GraphProvenance] = Field(default_factory=list)


class EntityNode(BaseModel):
    node_kind: Literal["entity"] = "entity"
    entity_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    entity_type: EntityType = "character"
    mention_count: int = 0
    first_seen_chapter: int = 0
    last_seen_chapter: int = 0
    episode_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationEdge(BaseModel):
    node_kind: Literal["relation"] = "relation"
    edge_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    validity_start_chapter: int
    validity_end_chapter: int | None = None
    episode_ids: list[str] = Field(default_factory=list)
    weight: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[GraphProvenance] = Field(default_factory=list)


class CommunityNode(BaseModel):
    node_kind: Literal["community"] = "community"
    community_id: str
    label: str
    entity_ids: list[str] = Field(default_factory=list)
    episode_ids: list[str] = Field(default_factory=list)
    chapter_start: int = 0
    chapter_end: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[GraphProvenance] = Field(default_factory=list)


class SagaNode(BaseModel):
    node_kind: Literal["saga"] = "saga"
    saga_id: str
    label: str
    episode_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    chapter_start: int = 0
    chapter_end: int = 0
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[GraphProvenance] = Field(default_factory=list)


class GraphStats(BaseModel):
    chapter_count: int = 0
    episode_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    community_count: int = 0
    saga_count: int = 0
    max_chapter_index: int = 0
    top_entities: list[dict[str, Any]] = Field(default_factory=list)
    entity_type_breakdown: dict[str, int] = Field(default_factory=dict)
    chapter_density: list[dict[str, int]] = Field(default_factory=list)


class TemporalContextGraph(BaseModel):
    graph_id: str = ""
    book_id: str
    title: str
    graph_version: str = "0.2.0"
    chapters: dict[str, ChapterNode] = Field(default_factory=dict)
    episodes: dict[str, EpisodeNode] = Field(default_factory=dict)
    entities: dict[str, EntityNode] = Field(default_factory=dict)
    relations: dict[str, RelationEdge] = Field(default_factory=dict)
    communities: dict[str, CommunityNode] = Field(default_factory=dict)
    sagas: dict[str, SagaNode] = Field(default_factory=dict)
    chapter_timeline: list[ChapterTimelineEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.graph_id = self.graph_id or f"graph::{self.book_id}"
        self.chapters = NodeTable(self.chapters)
        self.episodes = NodeTable(self.episodes)
        self.entities = NodeTable(self.entities)
        self.relations = NodeTable(self.relations)
        self.communities = NodeTable(self.communities)
        self.sagas = NodeTable(self.sagas)

    def stats(self) -> GraphStats:
        chapter_density = []
        for item in self.chapter_timeline:
            chapter_density.append(
                {
                    "chapter_index": item.chapter_index,
                    "episode_count": len(item.episode_ids),
                    "entity_count": len(item.entity_ids),
                    "relation_count": len(item.relation_ids),
                }
            )
        entity_counter = Counter(entity.entity_type for entity in self.entities.values())
        top_entities = [
            {
                "entity_id": entity.entity_id,
                "canonical_name": entity.canonical_name,
                "mention_count": entity.mention_count,
                "entity_type": entity.entity_type,
            }
            for entity in sorted(self.entities.values(), key=lambda item: item.mention_count, reverse=True)[:5]
        ]
        max_chapter = max((item.chapter_index for item in self.chapter_timeline), default=0)
        return GraphStats(
            chapter_count=len(self.chapters),
            episode_count=len(self.episodes),
            entity_count=len(self.entities),
            relation_count=len(self.relations),
            community_count=len(self.communities),
            saga_count=len(self.sagas),
            max_chapter_index=max_chapter,
            top_entities=top_entities,
            entity_type_breakdown=dict(entity_counter),
            chapter_density=chapter_density,
        )

    def visible_episode_ids(self, max_chapter: int | None = None) -> list[str]:
        return [
            episode_id
            for episode_id, episode in self.episodes.items()
            if max_chapter is None or episode.chapter_index <= max_chapter
        ]

    def chapter_window(self, start_chapter: int | None = None, end_chapter: int | None = None) -> list[ChapterTimelineEntry]:
        return [
            item
            for item in self.chapter_timeline
            if (start_chapter is None or item.chapter_index >= start_chapter)
            and (end_chapter is None or item.chapter_index <= end_chapter)
        ]

    def entity_neighbors(self, entity_id: str, relation_types: Iterable[str] | None = None) -> list[dict[str, Any]]:
        allowed = set(relation_types or [])
        neighbors: list[dict[str, Any]] = []
        for relation in self.relations.values():
            if relation.source_entity_id != entity_id and relation.target_entity_id != entity_id:
                continue
            if allowed and relation.relation_type not in allowed:
                continue
            neighbor_id = relation.target_entity_id if relation.source_entity_id == entity_id else relation.source_entity_id
            neighbor = self.entities.get(neighbor_id)
            if neighbor is None:
                continue
            neighbors.append(
                {
                    "entity_id": neighbor.entity_id,
                    "canonical_name": neighbor.canonical_name,
                    "entity_type": neighbor.entity_type,
                    "relation_type": relation.relation_type,
                    "weight": relation.weight,
                    "validity_start_chapter": relation.validity_start_chapter,
                    "validity_end_chapter": relation.validity_end_chapter,
                }
            )
        return sorted(neighbors, key=lambda item: (item["weight"], item["canonical_name"]), reverse=True)

    def relation_lookup(self, source_entity_id: str, target_entity_id: str) -> list[RelationEdge]:
        matched: list[RelationEdge] = []
        for relation in self.relations.values():
            endpoints = {relation.source_entity_id, relation.target_entity_id}
            if {source_entity_id, target_entity_id} == endpoints:
                matched.append(relation)
        return sorted(matched, key=lambda item: (item.validity_start_chapter, item.weight))

    def browse(self, node_kind: NodeKind, limit: int = 20, max_chapter: int | None = None) -> list[BaseModel]:
        collection = getattr(self, f"{node_kind}s" if node_kind != "community" else "communities")
        items = list(collection.values())
        if max_chapter is not None:
            filtered: list[BaseModel] = []
            for item in items:
                chapter_index = getattr(item, "chapter_index", None)
                if chapter_index is None:
                    chapter_index = getattr(item, "chapter_start", None)
                if chapter_index is None or chapter_index <= max_chapter:
                    filtered.append(item)
            items = filtered
        return items[:limit]


class GraphQuery(BaseModel):
    query: str = ""
    max_chapter: int | None = None
    min_chapter: int | None = None
    top_k: int = 5
    entity_names: list[str] = Field(default_factory=list)
    entity_types: list[EntityType] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    node_types: list[NodeKind] = Field(default_factory=list)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    min_relation_weight: float = 0.0
    min_entity_mentions: int = 0
    include_chapters: bool = True
    include_entities: bool = True
    include_relations: bool = True
    include_communities: bool = True
    include_sagas: bool = True


class GraphHit(BaseModel):
    hit_id: str
    hit_type: NodeKind
    score: float
    reason: str
    chapter_index: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    provenance: list[GraphProvenance] = Field(default_factory=list)


class GraphRetrievalResult(BaseModel):
    query: GraphQuery
    hits: list[GraphHit] = Field(default_factory=list)
    visible_episode_count: int = 0
    visible_entity_count: int = 0
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    hit_type_breakdown: dict[str, int] = Field(default_factory=dict)
    graph_metadata: dict[str, Any] = Field(default_factory=dict)
    graph_stats: GraphStats = Field(default_factory=GraphStats)
