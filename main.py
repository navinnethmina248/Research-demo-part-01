"""
Run the full 4-stage demo pipeline and print a readable summary for a supervisor demo.

Usage:
    python main.py
"""

from src.pipeline import run_pipeline


def _line(char="-", n=70):
    print(char * n)


def main():
    print("Running IT4010 (J26-IT-417) proof-of-concept pipeline...\n")
    report = run_pipeline()

    _line("=")
    print("STAGE 1 — Extraction: videos processed")
    _line()
    for video in report["stage1_extraction"]:
        n_segments = len(video["segments"])
        print(f"  {video['video_id']}: {n_segments} segments extracted")

    _line("=")
    print("STAGE 2 — Unified Knowledge Base (fused across all videos)")
    _line()
    stats = report["stage2_graph_stats"]
    print(f"  Graph: {stats['node_count']} fused concept nodes, {stats['edge_count']} co-occurrence edges")
    print("  Top fused concepts (cross-video ones marked with *):")
    for c in report["stage2_unified_knowledge_base"][:10]:
        marker = "*" if c["cross_video"] else " "
        print(f"   {marker} {c['concept']:<30} mentions={c['mention_count']}  sources={len(c['sources'])}")

    _line("=")
    print("STAGE 2b — Generated Multilingual Content")
    _line()
    content = report["stage2_generated_content"]
    print("  [EN] Summary:")
    print(f"       {content['en']['summary']}")
    print("  [SI] Summary (glossary demo):")
    print(f"       {content['si']['summary']}")

    _line("=")
    print("STAGE 3 — Audience Feedback Mining")
    _line()
    for vid, fb in report["stage3_feedback"].items():
        print(f"  {vid}:")
        print(f"    sentiment breakdown: {fb['sentiment_breakdown']}  (score={fb['overall_sentiment_score']})")
        if fb["content_requests"]:
            print(f"    content requests: {fb['content_requests']}")

    _line("=")
    print("STAGE 4 — Affective Audio Analysis + Recommendations")
    _line()
    for vid, ar in report["stage4_audio_recommendations"].items():
        print(f"  {vid}: audio_emotion={ar['audio_emotion']}, sentiment={ar['sentiment_score']}")
        print(f"    -> {ar['recommendation']}")

    _line("=")
    print("Full JSON report saved to outputs/pipeline_report.json")
    print("Knowledge graph exported to outputs/unified_knowledge_graph.graphml")
    print("(GraphML opens in Neo4j Browser, Gephi, or yEd for a visual demo)")


if __name__ == "__main__":
    main()
