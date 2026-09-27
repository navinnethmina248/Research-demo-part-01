"""
Stage 3 — Audience Feedback Mining & Content Improvement Recommendation
Maps to: IT23246928 (K.W.M.C.S. Wijekoon)

Lightweight lexicon-based sentiment classifier (no model download needed) applied to
sample audience comments, plus keyword-based "request" detection to surface concrete
content-improvement suggestions — a simplified stand-in for the fuller NLP sentiment
pipeline the production system would use.
"""

import re
from typing import List, Dict

POSITIVE_WORDS = {
    "great", "awesome", "love", "loved", "helpful", "clear", "excellent", "amazing",
    "good", "best", "thanks", "thank", "easy", "well", "nice", "perfect", "finally",
}
NEGATIVE_WORDS = {
    "confusing", "confused", "hard", "difficult", "boring", "slow", "bad", "worst",
    "unclear", "lost", "skip", "too fast", "hate", "annoying", "wrong",
}
REQUEST_PATTERNS = [
    r"please (explain|cover|add|do|make)",
    r"can you (explain|cover|make|do)",
    r"could you (explain|cover|make|do)",
    r"i wish",
    r"would be (great|nice|helpful) if",
    r"next time",
]


def score_sentiment(comment: str) -> str:
    lower = comment.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in lower)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def detect_requests(comment: str) -> List[str]:
    lower = comment.lower()
    hits = [pat for pat in REQUEST_PATTERNS if re.search(pat, lower)]
    return [comment] if hits else []


def mine_feedback(video_id: str, comments: List[str]) -> Dict:
    breakdown = {"positive": 0, "negative": 0, "neutral": 0}
    requests = []
    for c in comments:
        breakdown[score_sentiment(c)] += 1
        requests.extend(detect_requests(c))

    total = len(comments) or 1
    overall_score = (breakdown["positive"] - breakdown["negative"]) / total  # -1..1

    return {
        "video_id": video_id,
        "comment_count": len(comments),
        "sentiment_breakdown": breakdown,
        "overall_sentiment_score": round(overall_score, 2),
        "content_requests": requests,
    }
