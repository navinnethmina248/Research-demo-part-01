"""
Stage 1 — Multimodal Segmentation & Speech/Key-Information Extraction
Maps to: IT23241350 (A.M.B. Wanigasekara) — "Cross-Video Knowledge Extraction"

PRODUCTION VERSION would:
  - Run OpenAI Whisper (or a hosted ASR API) on raw audio/video -> time-aligned transcript
  - Run a NER model (e.g. spaCy en_core_web_trf) for people/orgs/locations/dates
  - Use an LLM to pull out key concepts, facts, definitions and relationships per segment

THIS DEMO starts from a plain-text transcript (i.e. "already transcribed") and uses
rule-based segmentation + keyphrase extraction, so the whole pipeline runs offline,
deterministically, and at zero API cost. Every output item is tagged with a video_id,
segment_id and fake timestamp, exactly as required by the TAF (Objective 1 output spec),
so downstream fusion can still track full provenance.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Use scikit-learn's vetted ~320-word English stopword list as the base, plus a
# handful of filler words specific to spoken video/lecture transcripts.
STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "today", "video", "videos", "lesson", "next", "welcome", "hey", "everyone",
    "thanks", "watching", "subscribe", "quick", "note", "finish", "wraps", "lets",
    "let's", "let", "session", "going", "look", "talk", "start", "starting", "wrap",
}
MAX_PHRASE_WORDS = 4


@dataclass
class Segment:
    video_id: str
    segment_id: str
    start_ts: str
    end_ts: str
    text: str
    entities: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)


def _format_ts(total_seconds: int) -> str:
    m, s = divmod(total_seconds, 60)
    return f"{m:02d}:{s:02d}"


def extract_entities(text: str) -> List[str]:
    """Naive proper-noun / named-entity spotter.

    Looks for sequences of 1-3 capitalized words that are not sentence-initial
    stopwords — a cheap stand-in for a real NER model, good enough to demonstrate
    that entities carry through to the fused knowledge graph.
    """
    candidates = re.findall(r"(?:[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){0,2})", text)
    entities = []
    for c in candidates:
        words = c.split()
        if words[0].lower() in STOPWORDS:
            continue
        if len(c) <= 2:
            continue
        if c not in entities:
            entities.append(c)
    return entities


def extract_key_phrases(text: str, top_n: int = 6) -> List[str]:
    """RAKE-style keyphrase extraction: split on stopwords/punctuation into
    candidate phrases, then rank by (word frequency in this segment) summed
    over the phrase's words. No external corpus/model needed.
    """
    clean = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    tokens = clean.split()
    word_freq: Dict[str, int] = {}
    for t in tokens:
        tl = t.lower()
        if tl in STOPWORDS or len(tl) < 3:
            continue
        word_freq[tl] = word_freq.get(tl, 0) + 1

    phrases = []
    current = []
    for t in tokens:
        tl = t.lower()
        if tl in STOPWORDS or len(tl) < 3:
            if current:
                phrases.append(current)
                current = []
        else:
            current.append(t)
    if current:
        phrases.append(current)

    scored = []
    seen = set()
    for phrase_words in phrases:
        if not phrase_words:
            continue
        # cap runaway phrases (long stopword-free runs) to a coherent length
        phrase_words = phrase_words[:MAX_PHRASE_WORDS]
        phrase = " ".join(phrase_words)
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        score = sum(word_freq.get(w.lower(), 0) for w in phrase_words)
        scored.append((phrase, score))

    scored.sort(key=lambda x: -x[1])
    return [p for p, _ in scored[:top_n]]


def segment_transcript(video_id: str, raw_text: str) -> List[Segment]:
    """Split a transcript into topical chunks. We treat each paragraph as one
    "segment" (a stand-in for a shot/topic-boundary detector), and assign
    monotonically increasing fake timestamps at ~40s per segment."""
    paragraphs = [p.strip() for p in raw_text.strip().split("\n\n") if p.strip()]
    segments = []
    t = 0
    seg_seconds = 40
    for i, para in enumerate(paragraphs):
        seg = Segment(
            video_id=video_id,
            segment_id=f"{video_id}_seg{i+1}",
            start_ts=_format_ts(t),
            end_ts=_format_ts(t + seg_seconds),
            text=para,
        )
        segments.append(seg)
        t += seg_seconds
    return segments


def process_video(video_id: str, raw_text: str) -> Dict:
    """Full Stage 1 output for one video, in the standardized JSON shape the
    TAF specifies as the handoff into Stage 2 (Knowledge Fusion)."""
    segments = segment_transcript(video_id, raw_text)
    for seg in segments:
        seg.entities = extract_entities(seg.text)
        seg.key_phrases = extract_key_phrases(seg.text)
    return {
        "video_id": video_id,
        "segments": [asdict(s) for s in segments],
    }
