"""Benchmark harness and report generator.

These are the pieces that turn measurements into published numbers, so their
arithmetic is worth testing: a wrong percentile or a mis-parsed result file
would silently corrupt the comparison table.
"""
from __future__ import annotations

import json

import pytest

from benchmarks import harness, report


# -------------------------------------------------------------------- harness
def test_percentiles():
    values = [float(i) for i in range(1, 101)]
    stats = harness.percentiles(values)
    assert stats["p50"] == pytest.approx(50.5)
    assert stats["p95"] == 96.0
    assert stats["p99"] == 100.0
    assert stats["mean"] == pytest.approx(50.5)


def test_percentiles_of_nothing_is_none_not_zero():
    assert harness.percentiles([]) == {"p50": None, "p95": None, "p99": None, "mean": None}


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        ("ReadTimeout: timed out", "timeout"),
        ("ConnectError: connection refused", "connection"),
        ("HTTPStatusError: 503 Service Unavailable", "unavailable"),
        ("HTTPStatusError: 500 Internal Server Error", "server_error"),
        ("ValueError: something else", "other"),
    ],
)
def test_error_classification(error, kind):
    assert harness.classify(error) == kind


def test_eval_set_is_stable_across_runs(tmp_path):
    for i in range(30):
        (tmp_path / f"p{i:03d}.png").write_bytes(bytes([i]))

    first = harness.pick_eval_set(tmp_path, count=20)
    second = harness.pick_eval_set(tmp_path, count=20)

    assert first == second           # same seed, same order, byte-identical inputs
    assert len(first) == 20
    assert len(set(first)) == 20     # no page selected twice


def test_eval_set_requires_a_corpus(tmp_path):
    with pytest.raises(SystemExit):
        harness.pick_eval_set(tmp_path)


async def test_run_level_drives_the_configured_concurrency(ocr_client, sample_png):
    level = await harness.run_level(ocr_client, [sample_png], concurrency=2, total=6, warmup=1)

    assert level["requests"] == 6
    assert level["errors"] == 0
    assert level["concurrency"] == 2
    assert level["ttft_s"]["p50"] is not None
    assert level["agg_tokens_per_s"] > 0
    assert level["pages_per_min"] > 0


async def test_failures_are_counted_and_classified(sample_png, monkeypatch):
    import httpx

    from ocr_serving.common.engine import OCRClient

    client = OCRClient(base_url="http://engine.test/v1", model="m", max_retries=0)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(503, json={})),
        base_url="http://engine.test/v1",
    )
    level = await harness.run_level(client, [sample_png], concurrency=2, total=4, warmup=0)
    await client.aclose()

    assert level["errors"] == 4
    assert level["error_kinds"] == {"unavailable": 4}
    assert level["ttft_s"]["p50"] is None


# --------------------------------------------------------------------- report
@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    return tmp_path


def _write_result(path, label: str, ttft: float, agg: float) -> None:
    (path / f"{label}.json").write_text(
        json.dumps({
            "label": label,
            "levels": [{
                "concurrency": 4,
                "ttft_s": {"p50": ttft, "p95": ttft + 0.1},
                "e2e_s": {"p50": 1.0, "p95": 1.5},
                "req_tokens_per_s_mean": 40.0,
                "agg_tokens_per_s": agg,
                "pages_per_min": 12.0,
                "errors": 0,
            }],
        }),
        encoding="utf-8",
    )


def test_serving_table_lists_every_stack_found(results_dir):
    _write_result(results_dir, "vllm", 0.4, 200.0)
    _write_result(results_dir, "ray_serve", 0.6, 260.0)

    table = report.serving_table()

    assert "vLLM (standalone)" in table and "Ray Serve (serve.llm)" in table
    assert table.count("\n") == 3          # header, separator, two rows


def test_headline_picks_the_winners(results_dir):
    _write_result(results_dir, "vllm", 0.4, 200.0)
    _write_result(results_dir, "ray_serve", 0.6, 260.0)

    headline = report.headline()

    assert "0.4s" in headline and "vLLM (standalone)" in headline
    assert "260.0 tok/s" in headline and "Ray Serve" in headline


def test_missing_measurements_render_as_dashes(results_dir):
    (results_dir / "ops_ledger.json").write_text(
        json.dumps({"_comment": "template", "vllm": {"moving_parts": 1, "notes": "one process"}}),
        encoding="utf-8",
    )

    table = report.ops_table()

    assert "—" in table                    # not the string "None"
    assert "_comment" not in table         # template comment key skipped
    assert "one process" in table


def test_empty_results_produce_placeholders(results_dir):
    block = report.build_block()
    assert "No serving results yet" in block
    assert block.startswith(report.BEGIN) and block.endswith(report.END)


def test_report_patches_only_between_markers(results_dir, tmp_path):
    _write_result(results_dir, "vllm", 0.4, 200.0)
    target = tmp_path / "comparison.md"
    target.write_text(
        f"# Title\n\nkeep me\n\n{report.BEGIN}\nstale\n{report.END}\n\ntrailing prose\n",
        encoding="utf-8",
    )

    import sys

    argv = sys.argv
    sys.argv = ["report.py", "--target", str(target)]
    try:
        report.main()
    finally:
        sys.argv = argv

    text = target.read_text(encoding="utf-8")
    assert "keep me" in text and "trailing prose" in text
    assert "stale" not in text
    assert "vLLM (standalone)" in text
