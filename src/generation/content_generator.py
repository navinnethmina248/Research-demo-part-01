"""
Stage 2 (continued) — Adaptive Multilingual Content Generation
Maps to: IT23241428 (B.M. Navin Nethmina) — YOUR component

Takes the unified knowledge base produced by knowledge_fusion.py and generates
learner-facing content — a summary and key-point flashcards — in English and Sinhala,
demonstrating the "single knowledge base -> many languages" idea from the TAF.

PRODUCTION SWAP POINT: `translate_to_sinhala()` below is a tiny offline glossary,
used only so this demo needs no internet access or API key. Swap it for:
  - an open-source, self-hostable MT model such as NLLB-200 (facebook/nllb-200-distilled-600M
    via HuggingFace transformers — free, runs locally, no per-call cost), or
  - a hosted translation API if budget allows.
The glossary approach is NOT production-quality translation — it's here purely to prove
the pipeline's shape end-to-end without incurring API cost during development.
"""

from typing import List, Dict

# A small demo glossary — real system would use a proper MT model (see docstring above)
GLOSSARY_EN_SI = {
    "machine learning": "යන්ත්‍ර ඉගෙනීම",
    "neural network": "ස්නායු ජාලය",
    "neural networks": "ස්නායු ජාල",
    "deep learning": "ගැඹුරු ඉගෙනීම",
    "supervised learning": "අධීක්ෂණය කළ ඉගෙනීම",
    "unsupervised learning": "අධීක්ෂණය නොකළ ඉගෙනීම",
    "reinforcement learning": "ශක්තිමත් කිරීමේ ඉගෙනීම",
    "decision tree": "තීරණ ගස",
    "clustering": "පොකුරු කිරීම",
    "classification": "වර්ගීකරණය",
    "artificial intelligence": "කෘත්‍රිම බුද්ධිය",
}


def translate_to_sinhala(text: str) -> str:
    """Glossary-based term substitution — demo-only stand-in for a real MT model
    (see module docstring). Falls back to appending an untranslated tag when no
    glossary entry matches, so the demo output is honest about its limitation."""
    lower = text.lower()
    translated_terms = []
    for term, si in GLOSSARY_EN_SI.items():
        if term in lower:
            translated_terms.append(f"{term} → {si}")
    if translated_terms:
        return f"[Sinhala terms found] {'; '.join(translated_terms)}"
    return "[no glossary match — production would call an MT model here]"


def generate_summary(unified_kb: List[Dict], top_n: int = 8) -> Dict:
    """Produce an extractive-style summary and key-point list in English, plus a
    Sinhala rendering, from the fused knowledge base (highest cross-video /
    highest-mention concepts first)."""
    top_concepts = unified_kb[:top_n]
    key_points_en = [c["concept"] for c in top_concepts]
    cross_video_flags = [c["cross_video"] for c in top_concepts]

    summary_en = (
        "This content library covers: " + "; ".join(key_points_en) + ". "
        + f"{sum(cross_video_flags)} of these {len(key_points_en)} concepts are "
        "confirmed across more than one video, indicating strong topical overlap "
        "in the library."
    )

    return {
        "en": {
            "summary": summary_en,
            "key_points": key_points_en,
        },
        "si": {
            "summary": translate_to_sinhala(summary_en),
            "key_points": [translate_to_sinhala(p) for p in key_points_en],
        },
    }
