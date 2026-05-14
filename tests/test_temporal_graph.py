from pathlib import Path

from services.graph.builder import build_temporal_graph
from services.graph.models import GraphQuery
from services.graph.retrieval import TemporalGraphRetriever, search_temporal_graph
from services.graph.storage import load_graph, save_graph
from services.ingest.parser import build_book_record


def demo_graph():
    source = Path("examples/muse_demo_book.txt")
    record = build_book_record("muse_demo_book", source.read_text(encoding="utf-8"), source)
    return build_temporal_graph(record)


def test_temporal_graph_builds_richer_topology():
    graph = demo_graph()

    assert graph.graph_id.startswith("graph::")
    assert len(graph.chapters) >= 2
    assert len(graph.episodes) >= 6
    assert len(graph.sagas) >= 1
    assert any(entity.canonical_name == "Aya" for entity in graph.entities.values())
    assert any(episode.entities for episode in graph.episodes.values())
    assert any(saga.episode_ids for saga in graph.sagas.values())
    assert len(graph.chapter_timeline) == len(graph.chapters)


def test_temporal_graph_supports_browse_and_lookup_utilities():
    graph = demo_graph()

    browsed = graph.browse("episode", limit=2, max_chapter=1)
    assert len(browsed) == 2
    assert all(item.chapter_index <= 1 for item in browsed)

    aya = next(entity for entity in graph.entities.values() if entity.canonical_name == "Aya")
    neighbors = graph.entity_neighbors(aya.entity_id)
    assert isinstance(neighbors, list)
    if neighbors:
        assert "relation_type" in neighbors[0]

    chapter_head = graph.chapters[:1]
    assert len(chapter_head) == 1
    assert chapter_head[0].node_kind == "chapter"


def test_temporal_graph_search_respects_progress_boundary_and_filters():
    graph = demo_graph()
    hits = search_temporal_graph(graph, "Aya relationship question", max_chapter=1, top_k=10)
    assert hits
    assert all(hit.chapter_index <= 1 for hit in hits)

    retrieval = TemporalGraphRetriever().retrieve(
        graph,
        GraphQuery(
            query="Aya",
            max_chapter=2,
            top_k=8,
            entity_names=["Aya"],
            node_types=["episode", "entity", "chapter", "relation"],
            min_entity_mentions=1,
        ),
    )
    assert retrieval.hits
    assert retrieval.hit_type_breakdown["episode"] >= 1
    assert retrieval.graph_stats.chapter_count >= 2
    assert "max_chapter" in retrieval.applied_filters


def test_temporal_graph_storage_persists_metadata(tmp_path, monkeypatch):
    import services.graph.storage as graph_storage

    graph = demo_graph()
    monkeypatch.setattr(graph_storage, "GRAPHS_DIR", tmp_path)

    save_graph(graph)
    loaded = load_graph(graph.book_id)

    assert loaded.graph_id == graph.graph_id
    assert loaded.metadata["storage"]["storage_version"] == graph_storage.STORAGE_VERSION
    assert loaded.metadata["storage"]["loaded"] is True
    assert loaded.metadata["graph_stats"]["chapter_count"] >= 2
