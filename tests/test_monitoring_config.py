"""Monitoring configuration is code, and nothing was checking it.

A typo in an alert expression, a panel pointing at a metric that no longer
exists, or a scrape job with no targets all fail silently: Prometheus logs a
parse error nobody reads, or the panel simply draws an empty chart. These tests
parse the files and cross-check every ``ocr_`` series they mention against the
metrics the code actually registers.

CI additionally runs ``promtool check rules`` for the parts only Prometheus can
judge (expression syntax, duration formats).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

MONITORING = Path(__file__).resolve().parents[1] / "deploy" / "monitoring"
DASHBOARD = MONITORING / "grafana" / "provisioning" / "dashboards" / "ocr-pipeline.json"

#: Suffixes Prometheus appends to histogram and counter series.
DERIVED_SUFFIXES = ("_bucket", "_sum", "_count", "_total", "_created", "_info")


@pytest.fixture(scope="module")
def prometheus_config() -> dict:
    return yaml.safe_load((MONITORING / "prometheus.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def alert_rules() -> dict:
    return yaml.safe_load((MONITORING / "alerts.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def registered_metric_names() -> set[str]:
    """Every ocr_ series the code can emit, from the live Prometheus registry."""
    from prometheus_client import REGISTRY

    import ocr_serving.common.metrics  # noqa: F401  (registers the collectors)

    names = set()
    for collector in list(REGISTRY._collector_to_names.values()):
        names.update(collector)
    return {name for name in names if name.startswith("ocr_")}


def referenced_metric_names(text: str) -> set[str]:
    return set(re.findall(r"\bocr_[a-z0-9_]+", text))


def base_name(metric: str) -> str:
    for suffix in DERIVED_SUFFIXES:
        if metric.endswith(suffix):
            return metric[: -len(suffix)]
    return metric


# --------------------------------------------------------------- prometheus
def test_prometheus_config_scrapes_both_services(prometheus_config):
    jobs = {job["job_name"] for job in prometheus_config["scrape_configs"]}
    assert {"gateway", "worker", "dcgm"} <= jobs


def test_every_scrape_job_has_targets(prometheus_config):
    for job in prometheus_config["scrape_configs"]:
        targets = [t for sc in job.get("static_configs", []) for t in sc.get("targets", [])]
        assert targets, f"scrape job {job['job_name']} has no targets"


def test_alert_rules_are_wired_into_prometheus(prometheus_config):
    assert prometheus_config["rule_files"] == ["/etc/prometheus/alerts.yml"]


# -------------------------------------------------------------------- alerts
def test_alert_rules_are_well_formed(alert_rules):
    groups = alert_rules["groups"]
    assert groups, "no alert groups defined"

    for group in groups:
        assert group.get("rules"), f"group {group.get('name')} has no rules"
        for rule in group["rules"]:
            name = rule.get("alert")
            assert name, f"a rule in {group['name']} has no alert name"
            assert str(rule.get("expr", "")).strip(), f"{name} has an empty expression"
            assert rule["labels"]["severity"] in {"warning", "critical"}, name
            annotations = rule.get("annotations", {})
            assert annotations.get("summary"), f"{name} has no summary"
            assert annotations.get("description"), f"{name} has no description"


def test_alert_names_are_unique(alert_rules):
    names = [r["alert"] for g in alert_rules["groups"] for r in g["rules"]]
    assert len(names) == len(set(names))


def test_the_failure_modes_that_matter_are_alerted(alert_rules):
    names = {r["alert"] for g in alert_rules["groups"] for r in g["rules"]}
    assert {"GatewayDown", "NoWorkerRunning", "QueueBacklogGrowing", "JobsDeadLettered"} <= names


def test_alerts_only_reference_metrics_the_code_emits(alert_rules):
    known = registered_metric_names()
    referenced = referenced_metric_names(yaml.safe_dump(alert_rules))
    assert known and referenced, "guard: this check is worthless if either set is empty"

    unknown = {m for m in referenced if base_name(m) not in known}
    assert not unknown, f"alerts reference metrics that are never emitted: {sorted(unknown)}"


# ----------------------------------------------------------------- dashboard
def panels(dashboard: dict) -> list[dict]:
    return [p for p in dashboard["panels"] if p.get("type") != "row"]


def test_dashboard_has_an_identity(dashboard):
    assert dashboard["uid"] and dashboard["title"]
    assert dashboard["schemaVersion"] >= 36


def test_every_panel_queries_the_provisioned_datasource(dashboard):
    for panel in panels(dashboard):
        source = panel.get("datasource", {})
        assert source.get("uid") == "prometheus", f"{panel['title']} has no datasource"
        assert panel.get("targets"), f"{panel['title']} has no query"
        for target in panel["targets"]:
            assert target.get("expr", "").strip(), f"{panel['title']} has an empty expression"


def test_panel_titles_and_positions_are_sane(dashboard):
    titles = [p["title"] for p in panels(dashboard)]
    assert len(titles) == len(set(titles)), "duplicate panel titles"

    for panel in dashboard["panels"]:
        pos = panel["gridPos"]
        assert 0 <= pos["x"] < 24 and pos["x"] + pos["w"] <= 24, panel.get("title")


def test_dashboard_only_references_metrics_the_code_emits(dashboard):
    known = registered_metric_names()
    referenced = referenced_metric_names(json.dumps(dashboard))
    assert known and referenced, "guard: this check is worthless if either set is empty"

    unknown = {m for m in referenced if base_name(m) not in known}
    assert not unknown, f"dashboard panels query metrics that are never emitted: {sorted(unknown)}"


def test_dashboard_covers_the_pipeline_signals(dashboard):
    referenced = referenced_metric_names(json.dumps(dashboard))
    for metric in (
        "ocr_ttft_seconds_bucket",
        "ocr_queue_depth",
        "ocr_pages_total",
        "ocr_engine_tokens_total",
        "ocr_http_requests_total",
    ):
        assert metric in referenced, f"no panel shows {metric}"


def test_grafana_provisioning_points_at_this_dashboard():
    provisioning = yaml.safe_load(
        (MONITORING / "grafana" / "provisioning" / "dashboards" / "dashboards.yml")
        .read_text(encoding="utf-8")
    )
    assert provisioning["providers"][0]["options"]["path"].endswith("/dashboards")

    datasource = yaml.safe_load(
        (MONITORING / "grafana" / "provisioning" / "datasources" / "prometheus.yml")
        .read_text(encoding="utf-8")
    )
    assert datasource["datasources"][0]["uid"] == "prometheus", (
        "panels reference the datasource by uid; it must be pinned, not auto-generated"
    )


def test_availability_alerts_are_aggregated(alert_rules):
    """A scrape job lists both the compose service and host.docker.internal so the
    stack works either way — which means one target is always down. An alert on the
    bare `up` series fires permanently; it has to aggregate."""
    rules = {r["alert"]: r["expr"] for g in alert_rules["groups"] for r in g["rules"]}

    for name in ("GatewayDown", "NoWorkerRunning"):
        expr = rules[name]
        assert "max(up{" in expr, f"{name} must aggregate over targets, got: {expr}"
        assert not re.search(r"(?<!\()up\{job=\"\w+\"\}\s*==\s*0", expr), (
            f"{name} still alerts on an individual target: {expr}"
        )
