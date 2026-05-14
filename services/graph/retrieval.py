from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .models import GraphHit, GraphQuery, GraphRetrievalResult, TemporalContextGraph


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _text_score(query_tokens: list[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    overlap = sum(1 for token in query_tokens if token in text_tokens)
    return overlap / math.sqrt(len(text_tokens))


def _matches_metadata(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if metadata.get(key) != expected:
            return False
    return True


class TemporalGraphRetriever:
    """Retrieve progress-aware temporal context with graph filters and browse support."""

    def retrieve(self, graph: TemporalContextGraph, query: GraphQuery) -> GraphRetrievalResult:
        visible_episode_ids = graph.visible_episode_ids(query.max_chapter)
        visible_entity_ids = {
            entity_id
            for entity_id, entity in graph.entities.items()
            if (query.max_chapter is None or entity.first_seen_chapter <= query.max_chapter)
            and (query.min_chapter is None or entity.last_seen_chapter >= query.min_chapter)
        }
        query_tokens = _tokenize(query.query)
        requested_entity_terms = {item.strip().lower() for item in query.entity_names if item.strip()}
        requested_tags = set(query.tags)
        requested_node_types = set(query.node_types)

        episode_hits: list[GraphHit] = []
        if not requested_node_types or "episode" in requested_node_types:
            episode_hits = self._retrieve_episodes(
                graph=graph,
                query=query,
                query_tokens=query_tokens,
                visible_episode_ids=visible_episode_ids,
                requested_entity_terms=requested_entity_terms,
                requested_tags=requested_tags,
            )
        ranked_episode_hits = sorted(episode_hits, key=lambda item: item.score, reverse=True)[: query.top_k]
        output_hits = list(ranked_episode_hits)
        supporting_episode_ids = {hit.hit_id for hit in ranked_episode_hits if hit.hit_type == "episode"}

        if query.include_chapters and (not requested_node_types or "chapter" in requested_node_types):
            output_hits.extend(self._retrieve_chapters(graph, query, supporting_episode_ids, query_tokens))
        if query.include_entities and (not requested_node_types or "entity" in requested_node_types):
            output_hits.extend(
                self._retrieve_entities(graph, query, visible_entity_ids, supporting_episode_ids, query_tokens)
            )
        if query.include_relations and (not requested_node_types or "relation" in requested_node_types):
            output_hits.extend(
                self._retrieve_relations(
                    graph,
                    query,
                    visible_episode_ids,
                    visible_entity_ids,
                    supporting_episode_ids,
                    query_tokens,
                )
            )
        if query.include_communities and (not requested_node_types or "community" in requested_node_types):
            output_hits.extend(self._retrieve_communities(graph, query, supporting_episode_ids, query_tokens))
        if query.include_sagas and (not requested_node_types or "saga" in requested_node_types):
            output_hits.extend(self._retrieve_sagas(graph, query, supporting_episode_ids, query_tokens))

        output_hits = sorted(output_hits, key=lambda item: (item.score, -(item.chapter_index or 0)), reverse=True)
        hit_breakdown = Counter(hit.hit_type for hit in output_hits)
        return GraphRetrievalResult(
            query=query,
            hits=output_hits[: max(query.top_k, len(ranked_episode_hits)) + 8],
            visible_episode_count=len(visible_episode_ids),
            visible_entity_count=len(visible_entity_ids),
            applied_filters={
                "max_chapter": query.max_chapter,
                "min_chapter": query.min_chapter,
                "entity_names": query.entity_names,
                "entity_types": query.entity_types,
                "relation_types": query.relation_types,
                "tags": query.tags,
                "node_types": query.node_types,
                "metadata_filters": query.metadata_filters,
            },
            hit_type_breakdown=dict(hit_breakdown),
            graph_metadata=graph.metadata,
            graph_stats=graph.stats(),
        )

    def _retrieve_episodes(
        self,
        graph: TemporalContextGraph,
        query: GraphQuery,
        query_tokens: list[str],
        visible_episode_ids: list[str],
        requested_entity_terms: set[str],
        requested_tags: set[str],
    ) -> list[GraphHit]:
        hits: list[GraphHit] = []
        for episode_id in visible_episode_ids:
            episode = graph.episodes[episode_id]
            if query.min_chapter is not None and episode.chapter_index < query.min_chapter:
                continue
            if requested_tags and not requested_tags.intersection(episode.tags):
                continue
            if query.metadata_filters and not _matches_metadata(episode.metadata, query.metadata_filters):
                continue

            entity_bonus = 0.0
            if requested_entity_terms:
                entity_names = {
                    graph.entities[entity_id].canonical_name.lower()
                    for entity_id in episode.entities
                    if entity_id in graph.entities
                }
                matches = requested_entity_terms.intersection(entity_names)
                entity_bonus = 0.75 * len(matches)
                if not matches:
                    continue

            metadata_bonus = 0.25 * sum(
                1
                for key, value in query.metadata_filters.items()
                if episode.metadata.get(key) == value
            )
            tag_bonus = 0.2 * len(requested_tags.intersection(episode.tags))
            score = _text_score(query_tokens, episode.text) + entity_bonus + metadata_bonus + tag_bonus
            if score <= 0:
                continue

            hits.append(
                GraphHit(
                    hit_id=episode_id,
                    hit_type="episode",
                    score=round(score, 4),
                    reason="episode_text+entity+metadata",
                    chapter_index=episode.chapter_index,
                    payload={
                        "chunk_id": episode.chunk_id,
                        "entities": episode.entities,
                        "tags": episode.tags,
                        "community_ids": episode.community_ids,
                        "saga_ids": episode.saga_ids,
                        "spoiler_level": episode.spoiler_level,
                    },
                    provenance=episode.provenance,
                )
            )
        return hits

    def _retrieve_chapters(
        self,
        graph: TemporalContextGraph,
        query: GraphQuery,
        supporting_episode_ids: set[str],
        query_tokens: list[str],
    ) -> list[GraphHit]:
        hits: list[GraphHit] = []
        for chapter in graph.chapters.values():
            if query.max_chapter is not None and chapter.chapter_index > query.max_chapter:
                continue
            if query.min_chapter is not None and chapter.chapter_index < query.min_chapter:
                continue
            if not supporting_episode_ids.intersection(chapter.episode_ids):
                continue
            score = _text_score(query_tokens, f"{chapter.title} {chapter.metadata.get('timeline_summary', '')}")
            score += min(len(supporting_episode_ids.intersection(chapter.episode_ids)), 3) * 0.35
            if score <= 0:
                continue
            hits.append(
                GraphHit(
                    hit_id=chapter.chapter_node_id,
                    hit_type="chapter",
                    score=round(score, 4),
                    reason="chapter_timeline_overlap",
                    chapter_index=chapter.chapter_index,
                    payload={
                        "chapter_id": chapter.chapter_id,
                        "title": chapter.title,
                        "entity_ids": chapter.entity_ids,
                        "relation_ids": chapter.relation_ids,
                        "paragraph_count": chapter.paragraph_count,
                    },
                    provenance=chapter.provenance[:3],
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:3]

    def _retrieve_entities(
        self,
        graph: TemporalContextGraph,
        query: GraphQuery,
        visible_entity_ids: set[str],
        supporting_episode_ids: set[str],
        query_tokens: list[str],
    ) -> list[GraphHit]:
        hits: list[GraphHit] = []
        requested_entities = {value.lower() for value in query.entity_names}
        allowed_types = set(query.entity_types)
        for entity_id in visible_entity_ids:
            entity = graph.entities[entity_id]
            if entity.mention_count < query.min_entity_mentions:
                continue
            if allowed_types and entity.entity_type not in allowed_types:
                continue
            searchable_text = " ".join([entity.canonical_name, *entity.aliases, entity.entity_type])
            score = _text_score(query_tokens, searchable_text)
            if requested_entities and entity.canonical_name.lower() in requested_entities:
                score += 1.0
            support_overlap = supporting_episode_ids.intersection(entity.episode_ids)
            if support_overlap:
                score += min(len(support_overlap), 3) * 0.3
            if score <= 0:
                continue
            hits.append(
                GraphHit(
                    hit_id=entity.entity_id,
                    hit_type="entity",
                    score=round(score, 4),
                    reason="entity_name+episode_overlap",
                    chapter_index=entity.first_seen_chapter,
                    payload={
                        "canonical_name": entity.canonical_name,
                        "entity_type": entity.entity_type,
                        "mention_count": entity.mention_count,
                        "episode_ids": entity.episode_ids[:6],
                        "neighbor_count": len(graph.entity_neighbors(entity.entity_id)),
                    },
                    provenance=[
                        graph.episodes[episode_id].provenance[0]
                        for episode_id in entity.episode_ids[:3]
                        if episode_id in graph.episodes and graph.episodes[episode_id].provenance
                    ],
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:3]

    def _retrieve_relations(
        self,
        graph: TemporalContextGraph,
        query: GraphQuery,
        visible_episode_ids: list[str],
        visible_entity_ids: set[str],
        supporting_episode_ids: set[str],
        query_tokens: list[str],
    ) -> list[GraphHit]:
        hits: list[GraphHit] = []
        visible_episodes = set(visible_episode_ids)
        requested_entities = {value.lower() for value in query.entity_names}
        allowed_relation_types = set(query.relation_types)
        for edge in graph.relations.values():
            if edge.weight < query.min_relation_weight:
                continue
            if allowed_relation_types and edge.relation_type not in allowed_relation_types:
                continue
            if edge.source_entity_id not in visible_entity_ids or edge.target_entity_id not in visible_entity_ids:
                continue
            if query.max_chapter is not None and edge.validity_start_chapter > query.max_chapter:
                continue
            if query.min_chapter is not None and (edge.validity_end_chapter or 0) < query.min_chapter:
                continue
            if not visible_episodes.intersection(edge.episode_ids):
                continue

            source_name = graph.entities[edge.source_entity_id].canonical_name
            target_name = graph.entities[edge.target_entity_id].canonical_name
            name_text = f"{source_name} {target_name} {edge.relation_type}"
            score = _text_score(query_tokens, name_text) + min(edge.weight, 3.0) * 0.2
            if requested_entities and {source_name.lower(), target_name.lower()}.intersection(requested_entities):
                score += 0.8
            if supporting_episode_ids.intersection(edge.episode_ids):
                score += 0.6
            if score <= 0:
                continue
            hits.append(
                GraphHit(
                    hit_id=edge.edge_id,
                    hit_type="relation",
                    score=round(score, 4),
                    reason="edge_validity+entity_overlap",
                    chapter_index=edge.validity_start_chapter,
                    payload={
                        "source_entity_id": edge.source_entity_id,
                        "target_entity_id": edge.target_entity_id,
                        "relation_type": edge.relation_type,
                        "validity_start_chapter": edge.validity_start_chapter,
                        "validity_end_chapter": edge.validity_end_chapter,
                        "episode_ids": edge.episode_ids,
                        "weight": edge.weight,
                    },
                    provenance=edge.provenance[:3],
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:3]

    def _retrieve_communities(
        self,
        graph: TemporalContextGraph,
        query: GraphQuery,
        supporting_episode_ids: set[str],
        query_tokens: list[str],
    ) -> list[GraphHit]:
        hits: list[GraphHit] = []
        for community in graph.communities.values():
            if query.max_chapter is not None and community.chapter_start > query.max_chapter:
                continue
            if query.min_chapter is not None and community.chapter_end < query.min_chapter:
                continue
            if not supporting_episode_ids.intersection(community.episode_ids):
                continue
            label_score = _text_score(query_tokens, community.label)
            score = label_score + 0.3 * min(len(community.entity_ids), 4)
            hits.append(
                GraphHit(
                    hit_id=community.community_id,
                    hit_type="community",
                    score=round(score, 4),
                    reason="community_overlap",
                    chapter_index=community.chapter_start,
                    payload={
                        "label": community.label,
                        "entity_ids": community.entity_ids,
                        "episode_ids": community.episode_ids[:6],
                    },
                    provenance=community.provenance[:3],
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:2]

    def _retrieve_sagas(
        self,
        graph: TemporalContextGraph,
        query: GraphQuery,
        supporting_episode_ids: set[str],
        query_tokens: list[str],
    ) -> list[GraphHit]:
        hits: list[GraphHit] = []
        for saga in graph.sagas.values():
            if query.max_chapter is not None and saga.chapter_start > query.max_chapter:
                continue
            if query.min_chapter is not None and saga.chapter_end < query.min_chapter:
                continue
            if not supporting_episode_ids.intersection(saga.episode_ids):
                continue
            score = _text_score(query_tokens, f"{saga.label} {saga.summary}") + 0.25 * len(
                supporting_episode_ids.intersection(saga.episode_ids)
            )
            hits.append(
                GraphHit(
                    hit_id=saga.saga_id,
                    hit_type="saga",
                    score=round(score, 4),
                    reason="saga_temporal_context",
                    chapter_index=saga.chapter_start,
                    payload={
                        "label": saga.label,
                        "summary": saga.summary,
                        "chapter_start": saga.chapter_start,
                        "chapter_end": saga.chapter_end,
                        "entity_ids": saga.entity_ids,
                    },
                    provenance=saga.provenance[:3],
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:2]


def search_temporal_graph(
    graph: TemporalContextGraph,
    query: str,
    max_chapter: int,
    top_k: int = 5,
) -> list[GraphHit]:
    result = TemporalGraphRetriever().retrieve(
        graph,
        GraphQuery(query=query, max_chapter=max_chapter, top_k=top_k),
    )
    return result.hits
