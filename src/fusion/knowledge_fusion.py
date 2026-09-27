"""
Stage 2 — Semantic Knowledge Fusion
Maps to: IT23241428 (B.M. Navin Nethmina) — YOUR component: "Semantic Knowledge Fusion
& Multilingual Content Generation"

This is the core novelty claim from the TAF: instead of treating each video's extracted
concepts as an isolated per-video output, we fuse concepts ACROSS every video in the
library into a single, de-duplicated knowledge base — while every concept keeps full
provenance (which video, which segment, which timestamp it came from).

Pipeline:
  1. Collect every key phrase extracted in Stage 1, across all videos.
  2. Vectorize the phrases (a lightweight, offline stand-in for the semantic embeddings
     the full system would use).
  3. Cluster near-duplicate phrases across videos using cosine similarity, so "Neural
     Network" (video 1), "Neural Networks" (video 2) and "neural network" (video 3)
     collapse into ONE node instead of three.
  4. Build a unified knowledge graph (networkx here; Neo4j in production) with
     co-occurrence edges between concepts that appear in the same segment.
  5. Export the fused knowledge base as JSON + GraphML (GraphML imports directly into
     Neo4j Browser, Gephi, or yEd for the visual demo to your supervisor).

PRODUCTION SWAP POINTS (clearly marked below) are exactly where you'd plug in the real
components without changing the surrounding architecture — useful to point out in your
supervisor meeting as evidence the design is modular and extensible.
"""

from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

SIMILARITY_THRESHOLD = 0.55  # tune per real embedding model; TF-IDF char-ngrams needs a lower bar than sentence embeddings


def collect_concepts(video_results: List[Dict]) -> List[Tuple[str, str, str, str]]:
    """Flatten every (key_phrase, video_id, segment_id, timestamp) tuple out of
    the Stage 1 outputs for every video."""
    concepts = []
    for video in video_results:
        for seg in video["segments"]:
            for phrase in seg["key_phrases"]:
                concepts.append((phrase, video["video_id"], seg["segment_id"], seg["start_ts"]))
    return concepts


def embed_concepts(concept_texts: List[str]):
    """PRODUCTION SWAP POINT: replace this TF-IDF (character n-gram) vectorizer
    with real semantic embeddings, e.g.:

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/LaBSE')  # multilingual, free, runs locally
        return model.encode(concept_texts)

    TF-IDF over character n-grams is used here because it needs no model download
    and no GPU, and still catches near-duplicate phrasing ("Neural Network" vs
    "neural networks") well enough for a demo — but it will NOT catch synonyms
    with different wording (e.g. "K-Means" vs "centroid-based clustering"), which
    is exactly the gap a real embedding model closes.
    """
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    return vectorizer.fit_transform([c.lower() for c in concept_texts])


def _cluster_by_similarity(sim_matrix: np.ndarray, threshold: float) -> List[int]:
    """Complete-linkage hierarchical clustering on a similarity matrix.

    We deliberately avoid a naive pairwise union-find ("if A~B and B~C, merge
    A,B,C") because that is single-linkage clustering and chains unrelated
    items together through an intermediate neighbour. Complete linkage instead
    requires EVERY pair inside a cluster to be similar enough, which is the
    right behaviour for de-duplicating concepts.
    """
    n = sim_matrix.shape[0]
    if n == 1:
        return [1]
    dist = 1 - sim_matrix
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)  # guard tiny negative floats from cosine_similarity
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="complete")
    distance_threshold = 1 - threshold
    return list(fcluster(z, t=distance_threshold, criterion="distance"))


def _pick_canonical(items: List[Tuple[str, str, str, str]]) -> str:
    """Pick the most-representative phrasing for a fused cluster: the most
    frequent exact wording (case-insensitive), tie-broken by shortest surface
    form, so labels stay short and readable in the demo graph."""
    counts = Counter(x[0].lower() for x in items)
    top_count = counts.most_common(1)[0][1]
    tied = [text for text in counts if counts[text] == top_count]
    best_lower = min(tied, key=len)
    for text, *_ in items:
        if text.lower() == best_lower:
            return text
    return items[0][0]


def fuse_concepts(video_results: List[Dict]):
    """Cluster concepts across all videos and build the unified knowledge graph.

    Returns:
        graph: a networkx.Graph — nodes are fused concepts, edges are co-occurrence
        unified_kb: list[dict] — the de-duplicated knowledge base with provenance
    """
    concepts = collect_concepts(video_results)
    if not concepts:
        return nx.Graph(), []

    texts = [c[0] for c in concepts]
    vectors = embed_concepts(texts)
    sim_matrix = cosine_similarity(vectors.toarray() if hasattr(vectors, "toarray") else vectors)

    labels = _cluster_by_similarity(sim_matrix, SIMILARITY_THRESHOLD)

    clusters: Dict[int, List[Tuple[str, str, str, str]]] = {}
    for i, c in enumerate(concepts):
        clusters.setdefault(labels[i], []).append(c)

    graph = nx.Graph()
    unified_kb = []
    # segment_id -> list of node_ids mentioned in that segment (for co-occurrence edges)
    segment_to_nodes: Dict[str, List[str]] = {}

    for idx, (root, items) in enumerate(clusters.items()):
        canonical = _pick_canonical(items)
        sources = [
            {"video_id": v, "segment_id": s, "timestamp": t} for _, v, s, t in items
        ]
        distinct_videos = sorted({v for _, v, _, _ in items})
        node_id = f"concept_{idx}"
        graph.add_node(
            node_id,
            label=canonical,
            sources=sources,
            mention_count=len(items),
            video_count=len(distinct_videos),
        )
        unified_kb.append(
            {
                "id": node_id,
                "concept": canonical,
                "mention_count": len(items),
                "cross_video": len(distinct_videos) > 1,
                "sources": sources,
            }
        )
        for _, _, seg_id, _ in items:
            segment_to_nodes.setdefault(seg_id, []).append(node_id)

    # co-occurrence edges: two concepts that showed up in the same segment are related
    for seg_id, node_ids in segment_to_nodes.items():
        unique_nodes = sorted(set(node_ids))
        for i in range(len(unique_nodes)):
            for j in range(i + 1, len(unique_nodes)):
                a, b = unique_nodes[i], unique_nodes[j]
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, weight=1)

    # sort so the most cross-video, most-mentioned concepts are easy to eyeball in a demo
    unified_kb.sort(key=lambda c: (-c["cross_video"], -c["mention_count"]))
    return graph, unified_kb


def export_graphml(graph: nx.Graph, path: str) -> None:
    """Export for visual inspection in Neo4j Browser / Gephi / yEd — good for a
    screenshot in your supervisor presentation."""
    # GraphML can't serialize list/dict node attributes, so flatten sources to a string first
    g2 = graph.copy()
    for _, data in g2.nodes(data=True):
        if "sources" in data:
            data["sources"] = "; ".join(
                f"{s['video_id']}@{s['timestamp']}" for s in data["sources"]
            )
    nx.write_graphml(g2, path)
