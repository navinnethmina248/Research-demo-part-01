# AI-Powered Knowledge Extraction & Content Generation Framework — Proof of Concept

**Project ID:** J26-IT-417 | SLIIT IT4010, CoEAI research group
**Topic:** AI-Powered Knowledge Extraction and Content Generation Framework for Videos and Podcasts for Education

This is a runnable, end-to-end **proof-of-concept** of the four-stage pipeline described
in the Topic Assessment Form. It exists to give your supervisor something concrete to
react to — a working skeleton that proves the architecture holds together — **not** a
production system. Every simplification is called out explicitly (see "What's real vs.
simulated" below) so nobody mistakes the demo shortcuts for the actual research claims.

## Architecture

```
Multiple Videos & Podcasts (sample transcripts)
        │
        ▼
Stage 1 — Multimodal Segmentation & Extraction        [IT23241350, Wanigasekara]
        │   (transcript → segments → entities + key phrases, per-segment)
        ▼
Stage 2 — Cross-Video Knowledge Fusion                [IT23241428, Navin — YOUR component]
   &     Adaptive Multilingual Content Generation
        │   (de-duplicate concepts across videos → unified knowledge graph
        │    → EN/SI summary + key points)
        ▼
Stage 3 — Audience Feedback Mining                    [IT23246928, Wijekoon]
        │   (sentiment + content-request detection from comments)
        ▼
Stage 4 — Affective Audio Analysis                    [IT23256828, Jayalath]
        │   (audio-emotion features ↔ correlated with Stage 3 sentiment)
        ▼
Final report: outputs/pipeline_report.json + outputs/unified_knowledge_graph.graphml
```

## How to run (in VS Code)

1. Open this folder in VS Code (`File > Open Folder...`).
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the demo:
   ```bash
   python main.py
   ```
4. Check `outputs/pipeline_report.json` for the full structured output, and open
   `outputs/unified_knowledge_graph.graphml` in **Neo4j Browser**, **Gephi**, or **yEd**
   for a visual of the fused knowledge graph — this is a strong visual to bring to your
   supervisor meeting.

No API keys, no GPU, no internet connection needed — it runs anywhere, which also keeps
it at **zero cost** while you iterate on the design.

## What's real vs. simulated in this demo

| Stage | This demo | Full production version |
|---|---|---|
| 1. Extraction | Rule-based segmentation + RAKE-style keyphrase extraction on plain-text transcripts | Whisper ASR on real audio/video, spaCy/LLM-based NER and concept extraction |
| 2. Fusion | TF-IDF (character n-grams) + complete-linkage clustering, networkx graph | Sentence-BERT / LaBSE embeddings, Neo4j graph database |
| 2. Generation | Extractive summary; Sinhala via a small hard-coded glossary | LLM-based abstractive summarization; NLLB-200 or similar MT model for real translation |
| 3. Feedback | Lexicon-based sentiment (small built-in word lists) | Trained sentiment/emotion classifier on real scraped comments |
| 4. Audio | Simulated features (deterministic, seeded by video ID — no real audio used) | librosa feature extraction + trained audio-emotion model on separated background music |

Every "production swap point" is marked with a comment in the corresponding source file
(`src/fusion/knowledge_fusion.py` especially, since that's your component) explaining
exactly what to replace and why the demo doesn't use it yet — mainly to avoid API cost
and GPU/model-download requirements while the architecture itself is being validated.

## Project structure

```
edu-ai-pipeline/
├── data/sample_transcripts/     # 3 sample "video" transcripts with deliberate topical overlap
├── src/
│   ├── extraction/               # Stage 1
│   ├── fusion/                   # Stage 2 — knowledge fusion (your component)
│   ├── generation/                # Stage 2 — multilingual generation (your component)
│   ├── feedback/                  # Stage 3
│   ├── audio/                     # Stage 4
│   └── pipeline.py                # orchestrates all 4 stages
├── main.py                        # CLI entry point — run this
├── requirements.txt
└── outputs/                       # generated on each run (gitignored)
```

## Why the sample data is designed this way

The three sample transcripts (`data/sample_transcripts/`) are about the same broad topic
(intro ML / deep learning) from three different "creators," with **deliberately
overlapping concepts** phrased differently ("Neural Network" vs "neural networks" vs
"Neural Networks") and each with some unique content too. This is specifically so you can
demonstrate the core novelty claim from the TAF — that concepts get fused into ONE node
across videos rather than staying siloed — and see it working when you run `main.py`.

## Talking points for your supervisor meeting

- Show the console output stage-by-stage — it mirrors the four sub-objectives in the TAF's
  Objectives table almost 1:1, which is useful for demonstrating traceability back to the
  proposal.
- Open the GraphML export visually — a picture of the fused knowledge graph is more
  convincing than a JSON blob.
- Be upfront that this is a lightweight architectural proof-of-concept: the value is in
  showing the four stages compose correctly and the data contracts between them (JSON
  shapes, provenance tracking) work, not in claiming these are the final algorithms.
- Good next step to propose: swap TF-IDF for `sentence-transformers` embeddings in your
  fusion module first (it's a 5-line change — see the comment in `embed_concepts()`) and
  compare de-duplication quality before/after, as a concrete "here's my next milestone."

## Cost-conscious design notes

Every library used here (`networkx`, `scikit-learn`, `scipy`) is free, open-source, and
runs entirely locally — no paid API calls anywhere in this demo. The production swap
points also favor free/self-hostable options (Sentence-BERT/LaBSE, NLLB-200) over paid
APIs, which is worth mentioning if cost is a factor your supervisor asks about for the
full implementation.
