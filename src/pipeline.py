"""
End-to-end orchestration of the four-stage pipeline described in the IT4010 TAF
(Project J26-IT-417, "AI-Powered Knowledge Extraction and Content Generation
Framework for Videos and Podcasts for Education").

Run via `python main.py` from the project root.
"""

import json
import os

from src.extraction.transcript_extractor import process_video
from src.fusion.knowledge_fusion import fuse_concepts, export_graphml
from src.generation.content_generator import generate_summary
from src.feedback.feedback_miner import mine_feedback
from src.audio.affective_audio import simulate_audio_features, correlate_with_feedback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "data", "sample_transcripts")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Mock audience comments per video, standing in for real scraped comments (Stage 3 input)
SAMPLE_COMMENTS = {
    "video1_intro_to_ml": [
        "This was so helpful, finally understand supervised vs unsupervised learning!",
        "Great intro, thanks!",
        "A bit confusing around decision trees, could you explain that part again?",
        "Please cover model evaluation metrics next time.",
    ],
    "video2_getting_started_ml": [
        "Loved the reinforcement learning explanation, very clear.",
        "The pacing was too fast for me, felt lost midway.",
        "Can you explain K-Means with a visual example next time?",
        "Nice video, easy to follow.",
    ],
    "video3_deep_learning_foundations": [
        "Backpropagation explanation was excellent!",
        "This is way too advanced, I wish there was a beginner version.",
        "CNNs finally make sense to me, thank you.",
        "Please add a diagram for RNNs, it was hard to visualize.",
    ],
}


def load_sample_transcripts():
    transcripts = {}
    for filename in sorted(os.listdir(TRANSCRIPT_DIR)):
        if filename.endswith(".txt"):
            video_id = filename[:-4]
            with open(os.path.join(TRANSCRIPT_DIR, filename), "r", encoding="utf-8") as f:
                transcripts[video_id] = f.read()
    return transcripts


def run_pipeline() -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transcripts = load_sample_transcripts()

    # ---- Stage 1: Extraction ----
    video_results = [process_video(vid, text) for vid, text in transcripts.items()]

    # ---- Stage 2: Cross-video knowledge fusion ----
    graph, unified_kb = fuse_concepts(video_results)
    export_graphml(graph, os.path.join(OUTPUT_DIR, "unified_knowledge_graph.graphml"))

    # ---- Stage 2b: Adaptive multilingual content generation ----
    generated_content = generate_summary(unified_kb)

    # ---- Stage 3: Audience feedback mining ----
    feedback_reports = {
        vid: mine_feedback(vid, SAMPLE_COMMENTS.get(vid, [])) for vid in transcripts
    }

    # ---- Stage 4: Affective audio analysis + correlation with Stage 3 ----
    audio_reports = {}
    for vid in transcripts:
        features = simulate_audio_features(vid)
        audio_reports[vid] = correlate_with_feedback(features, feedback_reports[vid])

    report = {
        "stage1_extraction": video_results,
        "stage2_unified_knowledge_base": unified_kb,
        "stage2_graph_stats": {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
        },
        "stage2_generated_content": generated_content,
        "stage3_feedback": feedback_reports,
        "stage4_audio_recommendations": audio_reports,
    }

    with open(os.path.join(OUTPUT_DIR, "pipeline_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report
