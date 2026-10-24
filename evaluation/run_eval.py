"""Evaluate retrieval (and optionally generation) on a question set.

Usage::

    python -m evaluation.run_eval                                   # tuning set
    python -m evaluation.run_eval --questions evaluation/heldout_questions.json
    python -m evaluation.run_eval --generate                        # also call the LLM

Writes a Markdown report to ``evaluation/results/`` (and, with
``--generate``, a JSON file of answers for manual review). Generated answers
are NOT scored automatically; a person must judge them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from chatbot.bootstrap import build_service, load_or_build_store
from chatbot.retriever import LanguageProfile, Retriever
from config.settings import PROJECT_ROOT, Settings, configure_logging
from knowledge_base.embeddings import SentenceTransformerEmbedder

DEFAULT_QUESTIONS = PROJECT_ROOT / "evaluation" / "eval_questions.json"
OUT_DIR = PROJECT_ROOT / "evaluation" / "results"
THRESHOLDS = [0.10, 0.14, 0.18, 0.22, 0.25, 0.30, 0.35]


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def retrieval_eval(settings: Settings, questions_path: Path) -> str:
    questions = _load(questions_path)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = load_or_build_store(settings, embedder)
    configured = {"en": settings.min_relevance_en, "ta": settings.min_relevance_ta}
    # Threshold and relative cutoff disabled so we can see the full ranking and sweep thresholds.
    unfiltered = Retriever(
        embedder,
        store,
        top_k=settings.top_k,
        profiles={
            "en": LanguageProfile(settings.dense_weight_en, -1.0),
            "ta": LanguageProfile(settings.dense_weight_ta, -1.0),
        },
        relative_cutoff=0.0,
    )
    # Exactly what the app sends to the LLM.
    configured_retriever = Retriever(
        embedder,
        store,
        top_k=settings.top_k,
        profiles={
            "en": LanguageProfile(settings.dense_weight_en, settings.min_relevance_en),
            "ta": LanguageProfile(settings.dense_weight_ta, settings.min_relevance_ta),
        },
        relative_cutoff=settings.relative_cutoff,
    )
    embedder.embed(["warm-up"])  # exclude model load time from latency

    rows = []
    for q in questions:
        start = time.perf_counter()
        passages = unfiltered.retrieve(q["question"], q["language"])
        latency = (time.perf_counter() - start) * 1000
        expected = set(q["expected_documents"])
        rank = next((i + 1 for i, p in enumerate(passages) if p.chunk.document_id in expected), None)
        relevant = next((p.score for p in passages if p.chunk.document_id in expected), None)
        sent = configured_retriever.retrieve(q["question"], q["language"])
        rows.append(
            {
                "id": q["id"],
                "language": q["language"],
                "in_scope": bool(expected),
                "rank": rank,
                "top1_doc": passages[0].chunk.document_id,
                "top1_score": passages[0].score,
                "relevant_score": relevant,
                "latency_ms": latency,
                "reaches_llm": any(p.chunk.document_id in expected for p in sent),
                "passages_sent": len(sent),
            }
        )

    ins = [r for r in rows if r["in_scope"]]
    oos = [r for r in rows if not r["in_scope"]]
    lines = [
        f"# Retrieval evaluation – `{questions_path.name}`",
        "",
        f"Embedding model: `{settings.embedding_model}` (language-centred) + char-4-gram TF-IDF; "
        f"chunks: {store.size}; top_k = {settings.top_k}; dense weight en={settings.dense_weight_en}, "
        f"ta={settings.dense_weight_ta}.",
        f"Questions: {len(ins)} in-scope ({sum(r['language'] == 'en' for r in ins)} English, "
        f"{sum(r['language'] == 'ta' for r in ins)} Tamil), {len(oos)} out-of-scope.",
        "",
        "## Ranking quality (threshold disabled)",
        "",
        "| Subset | n | Hit@1 | Hit@k | MRR@k |",
        "|---|---|---|---|---|",
    ]
    for name, subset in (
        ("All in-scope", ins),
        ("English", [r for r in ins if r["language"] == "en"]),
        ("Tamil", [r for r in ins if r["language"] == "ta"]),
    ):
        n = len(subset) or 1
        hit1 = sum(r["rank"] == 1 for r in subset) / n
        hitk = sum(r["rank"] is not None for r in subset) / n
        mrr = sum(1 / r["rank"] for r in subset if r["rank"]) / n
        lines.append(f"| {name} | {len(subset)} | {hit1:.2f} | {hitk:.2f} | {mrr:.2f} |")

    lines += [
        "",
        "## Relevance-threshold sweep (per question language)",
        "",
        "*Answerable* = in-scope question whose correct document is retrieved above the threshold. "
        "*Rejected* = out-of-scope question with no passage above the threshold (so the bot says "
        "verified information is unavailable).",
        "",
        "| Lang | Threshold | In-scope answerable | Out-of-scope rejected |",
        "|---|---|---|---|",
    ]
    for lang in ("en", "ta"):
        li = [r for r in ins if r["language"] == lang]
        lo = [r for r in oos if r["language"] == lang]
        for th in THRESHOLDS:
            ok_in = sum(1 for r in li if r["relevant_score"] is not None and r["relevant_score"] >= th)
            ok_oos = sum(1 for r in lo if r["top1_score"] < th)
            marker = " **(configured)**" if abs(th - configured[lang]) < 1e-9 else ""
            lines.append(f"| {lang} | {th:.2f}{marker} | {ok_in}/{len(li)} | {ok_oos}/{len(lo)} |")

    lines += [
        "",
        f"## With the configured settings (threshold + relative cutoff {settings.relative_cutoff})",
        "",
        "| Lang | Correct document reaches the LLM | Mean passages sent (in-scope) |",
        "|---|---|---|",
    ]
    for lang in ("en", "ta"):
        li = [r for r in ins if r["language"] == lang]
        mean_sent = statistics.mean(r["passages_sent"] for r in li) if li else 0.0
        lines.append(f"| {lang} | {sum(r['reaches_llm'] for r in li)}/{len(li)} | {mean_sent:.1f} |")

    lat = [r["latency_ms"] for r in rows]
    lines += [
        "",
        f"Retrieval latency per question (CPU, model already loaded): median {statistics.median(lat):.0f} ms, "
        f"max {max(lat):.0f} ms.",
        "",
        "## Per-question results",
        "",
        "| ID | Lang | Rank of correct doc | Top-1 document | Top-1 score | Correct-doc score | Passes threshold |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rel = f"{r['relevant_score']:.3f}" if r["relevant_score"] is not None else "–"
        rank = (r["rank"] if r["rank"] is not None else "not in top-k") if r["in_scope"] else "(out of scope)"
        passes = "yes" if r["top1_score"] >= configured[r["language"]] else "no"
        lines.append(
            f"| {r['id']} | {r['language']} | {rank} | {r['top1_doc']} | {r['top1_score']:.3f} | {rel} | {passes} |"
        )
    return "\n".join(lines) + "\n"


def generation_eval(settings: Settings, questions_path: Path) -> list[dict]:
    """Run the full pipeline and save outputs for manual review."""
    service = build_service(settings)
    outputs = []
    for q in _load(questions_path):
        resp = service.answer(q["question"], q["language"])
        outputs.append(
            {
                "id": q["id"],
                "question": q["question"],
                "language": q["language"],
                "status": resp.status,
                "mode": resp.mode,
                "latency_ms": resp.latency_ms,
                "answer": resp.answer,
                "warnings": resp.warnings,
                "cited_sources": [s.title for s in resp.sources if s.cited],
                "manual_review": {"supported_by_sources": None, "language_quality": None, "notes": ""},
            }
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="AgriBot AI evaluation")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--generate", action="store_true", help="also run the configured LLM provider")
    args = parser.parse_args()
    configure_logging()
    settings = Settings.from_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.questions.stem

    report = retrieval_eval(settings, args.questions)
    (OUT_DIR / f"retrieval_{stem}.md").write_text(report, encoding="utf-8")
    print(report)

    if args.generate:
        outputs = generation_eval(settings, args.questions)
        out = OUT_DIR / f"generation_{stem}_{settings.llm_provider}.json"
        out.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
        statuses: dict[str, int] = {}
        for o in outputs:
            statuses[o["status"]] = statuses.get(o["status"], 0) + 1
        print("Generation statuses:", statuses)
        print("Median latency:", statistics.median(o["latency_ms"] for o in outputs), "ms")
        print("Saved to", out, "- review answers manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
