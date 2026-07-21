"""Offline quality evaluation for the RAG system.

Runs a fixed question set and scores retrieval and generation separately —
because they fail differently and are fixed differently:
    retrieval hit-rate  did the expected source appear in the retrieved chunks?
    faithfulness        did the answer contain the required facts?
    refusal probe       does an out-of-corpus question get refused, not answered?

Run:  python evaluate.py
"""
import json
import pathlib
import time

from app.service import PROMPT, RagService

# A question the corpus cannot answer — verifies the grounding refusal holds.
REFUSAL_PROBE = "What is the capital of France?"


def run():
    cases = json.loads(pathlib.Path("eval_set.json").read_text())
    service = RagService()

    rows, hits, faithful, latencies = [], 0, 0, []
    print(f"Evaluating {len(cases)} questions on '{service.provider.name}' "
          f"(local models are slow on CPU — ~30s each)...\n", flush=True)
    for i, c in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {c['question'][:50]}...", flush=True)
        start = time.perf_counter()
        chunks = service.retrieval.search(c["question"], k=4)
        hit = c["expected_source"] in {ch.source for ch in chunks}

        context = "\n\n".join(ch.text for ch in chunks)
        answer = service.provider.generate(
            PROMPT.format(context=context, question=c["question"])
        )
        latencies.append(time.perf_counter() - start)

        ok = all(kw.lower() in answer.lower() for kw in c["must_contain"])
        hits += hit
        faithful += ok
        rows.append((c["question"][:48], hit, ok))

    refusal = service.answer(REFUSAL_PROBE)
    refused = "don't have that information" in refusal.text.lower()

    n = len(cases)
    p50 = sorted(latencies)[len(latencies) // 2]
    lines = [
        f"# Evaluation ({service.provider.name})\n",
        f"- Retrieval hit-rate: {hits}/{n} ({hits / n * 100:.0f}%)",
        f"- Answer faithfulness: {faithful}/{n} ({faithful / n * 100:.0f}%)",
        f"- Refusal on out-of-corpus question: {'PASS' if refused else 'FAIL'}",
        f"- Median query latency: {p50:.2f}s\n",
        "| Question | Right source | Key facts |",
        "|---|:---:|:---:|",
        *[f"| {q} | {'✅' if h else '❌'} | {'✅' if o else '❌'} |" for q, h, o in rows],
    ]
    report = "\n".join(lines)
    print(report)
    pathlib.Path("eval_results.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    run()
