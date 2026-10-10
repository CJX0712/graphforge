"""离线兜底测试：证明 graphForge 完全不依赖 scipy / networkx / scikit-learn。

作者：晨星

两条防线：
1. 源码静态扫描：任何 .py 都不得 import 重依赖。
2. 子进程动态验证：在 import 拦截器中跑通整条链路（blocked 模块会直接抛 ImportError）。
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAVY = {"scipy", "networkx", "sklearn", "sklearn_extra", "scikit_learn"}


def test_source_has_no_heavy_imports():
    for f in ROOT.rglob("*.py"):
        if ".git" in f.parts:
            continue
        text = f.read_text(encoding="utf-8")
        for mod in HEAVY:
            assert f"import {mod}" not in text, f"{f} 引入了重依赖 {mod}"
            assert f"from {mod}" not in text, f"{f} 引入了重依赖 {mod}"


def test_runs_without_heavy_deps():
    """在拦截 scipy/networkx/sklearn 的环境下跑通 benchmark 的关键路径。"""
    code = """
import sys

class Blocker:
    BLOCK = {"scipy", "networkx", "sklearn", "scikit_learn"}
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCK:
            raise ImportError("blocked: " + name)
        return None

sys.meta_path.insert(0, Blocker())
sys.path.insert(0, ".")

from graphforge.core.config import Config
from graphforge.pipeline.pipeline import benchmark
from graphforge.graph.community import louvain, modularity
from graphforge.graph.spectral import spectral_clustering
from graphforge.graph.kernels import weisfeiler_lehman_kernel
from graphforge.graph.centrality import pagerank
from graphforge.graph.build import stochastic_block_model

cfg = Config()
b = benchmark(seeds=[0], regimes=["balanced_strong"], config=cfg)
print("OK", b["summary"]["fuse"]["ari_mean"])
"""
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"离线链路失败:\n{res.stdout}\n{res.stderr}"
    assert "OK" in res.stdout


def test_no_heavy_modules_loaded_after_import():
    import graphforge.pipeline.pipeline  # noqa: F401

    loaded = {m.split(".")[0] for m in sys.modules}
    assert "scipy" not in loaded
    assert "networkx" not in loaded
    assert "sklearn" not in loaded
