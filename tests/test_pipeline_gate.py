"""门禁与 CLI 冒烟测试。

作者：晨星
"""

import json
import subprocess
import sys
from pathlib import Path

from graphforge.core.config import Config, get_config
from graphforge.pipeline.pipeline import REGIMES, benchmark, run_one

ROOT = Path(__file__).resolve().parent.parent


def test_gate_all_pass():
    """四项门禁全绿（完整套件 3 regime × 5 seeds）。"""
    cfg = get_config()
    b = benchmark(config=cfg)
    gate = b["gate"]
    assert gate["G1_recovery_ari"]["pass"], gate["G1_recovery_ari"]
    assert gate["G2_margin_over_plain_louvain"]["pass"], gate["G2_margin_over_plain_louvain"]
    assert gate["G3_modularity_monotone"]["pass"], gate["G3_modularity_monotone"]
    assert gate["G4_invariants"]["pass"], gate["G4_invariants"]
    assert gate["all_pass"]


def test_benchmark_shapes():
    b = benchmark(seeds=[0], regimes=["balanced_strong"], config=Config())
    assert b["n_runs"] == 1
    for m in ["fuse", "louvain_plain", "spectral_kmeans", "kmeans_adjacency", "random"]:
        assert m in b["summary"]
        assert -1.0 <= b["summary"][m]["ari_mean"] <= 1.0


def test_run_one_structure():
    spec = REGIMES["balanced_strong"]
    r = run_one(0, spec["sizes"], spec["p_in"], spec["p_out"], Config(), regime="balanced_strong")
    assert r["regime"] == "balanced_strong"
    assert r["k"] == 4
    assert set(r["fuse"]) >= {"ari", "nmi", "vi", "modularity", "n_communities"}


def test_cli_benchmark_smoke(tmp_path):
    out = tmp_path / "b.json"
    res = subprocess.run(
        [sys.executable, "-m", "graphforge.cli", "benchmark", "--seeds", "0", "--out", str(out)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "gate" in data


def test_cli_karate_and_wl_and_pagerank():
    for args in (
        ["karate"],
        ["wl", "--n", "20"],
        ["pagerank", "--n", "20", "--p", "0.3"],
    ):
        res = subprocess.run(
            [sys.executable, "-m", "graphforge.cli", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"{args}: {res.stderr}"
        assert res.stdout.strip()


def test_demo_smoke(tmp_path):
    out = tmp_path / "benchmark.json"
    res = subprocess.run(
        [sys.executable, "examples/run_demo.py", str(out)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["gate"]["all_pass"] is True
    assert data["determinism_bit_identical"] is True
    assert len(data["failure_cases"]) >= 3
