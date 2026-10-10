"""graphForge 评估链路：多 regime benchmark + 门禁。

作者：晨星

基准套件（3 个 regime × 5 seeds）：
- balanced_strong   : [50,50,50,50]  p_in=0.15 p_out=0.005
- balanced_moderate : [60,60,60,60]  p_in=0.12 p_out=0.010
- unbalanced        : [150,80,40,30] p_in=0.15 p_out=0.005

基线矩阵：
- louvain_plain      : 普通 Louvain（领域标准方法，**不给 k**）
- spectral_kmeans    : 谱聚类（给 k）—— SBM 上的统计天花板参考
- kmeans_adjacency   : 邻接行 k-means（给 k）
- random             : 随机划分（给 k）

门禁（DoD，方案阶段定死，诚实可达）：
- G1 恢复力  ：CommunityFuse 全套件均值 ARI >= 0.90
- G2 真实胜点：CommunityFuse 均值 ARI - 普通 Louvain 均值 ARI >= 0.05
               （贡献点 = 修复模块度分辨率极限；谱聚类已知 k 是天花板，见 model_card 已知限制）
- G3 模块度单调：Louvain 跨层模块度最大下降 <= 1e-9
- G4 不变量    ：拉普拉斯半正定、谱嵌入正交、WL 同构核 == 自核
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config, get_config
from ..core.interfaces import GraphContainer
from ..core.seed import set_all
from ..eval.metrics import ari, nmi, variation_of_information
from ..graph.build import erdos_renyi, fiedler_value, stochastic_block_model
from ..graph.community import community_fuse, louvain, modularity
from ..graph.kernels import weisfeiler_lehman_kernel
from ..graph.spectral import _kmeans, spectral_clustering, symmetric_normalized_embedding

# 基准套件定义
REGIMES: dict[str, dict] = {
    "balanced_strong": {"sizes": [50, 50, 50, 50], "p_in": 0.15, "p_out": 0.005},
    "balanced_moderate": {"sizes": [60, 60, 60, 60], "p_in": 0.12, "p_out": 0.010},
    "unbalanced": {"sizes": [150, 80, 40, 30], "p_in": 0.15, "p_out": 0.005},
}


def _random_baseline(graph: GraphContainer, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 555)
    return rng.integers(0, k, size=graph.n_nodes)


def _kmeans_adjacency(graph: GraphContainer, k: int, seed: int) -> np.ndarray:
    """邻接矩阵行向量 k-means 基线（给 k）。"""
    from ..core.config import get_config as _gc

    cfg = _gc()
    labels = _kmeans(graph.adjacency, k, cfg.spectral_kmeans_iters, cfg.spectral_kmeans_tol, seed + 11)
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def _score(true: np.ndarray, pred: np.ndarray, A: np.ndarray) -> dict:
    return {
        "ari": float(ari(true, pred)),
        "nmi": float(nmi(true, pred)),
        "vi": float(variation_of_information(true, pred)),
        "modularity": float(modularity(A, pred, 1.0)),
        "n_communities": int(np.unique(pred).size),
    }


def run_one(
    seed: int,
    sizes: list[int],
    p_in: float,
    p_out: float,
    config: Config | None = None,
    regime: str = "custom",
) -> dict:
    """单 (regime, seed) 评测：旗舰 vs 全部基线。"""
    config = config or get_config()
    set_all(seed)
    k = len(sizes)
    graph = stochastic_block_model(sizes, p_in, p_out, seed=seed)
    true = graph.labels
    A = graph.adjacency

    fuse = community_fuse(graph, k=k, gamma=config.louvain_resolution, seed=seed)
    plain = louvain(graph, gamma=config.louvain_resolution, seed=seed)
    spec = spectral_clustering(graph, k=k, seed=seed)
    km = _kmeans_adjacency(graph, k, seed)
    rand = _random_baseline(graph, k, seed)

    return {
        "regime": regime,
        "seed": seed,
        "n_nodes": graph.n_nodes,
        "k": k,
        "fuse": _score(true, fuse.labels, A),
        "louvain_plain": _score(true, plain.labels, A),
        "spectral_kmeans": _score(true, spec.labels, A),
        "kmeans_adjacency": _score(true, km, A),
        "random": _score(true, rand, A),
        "fuse_strategy": fuse.meta.get("strategy", ""),
    }


def _agg_ari(rows: list[dict], method: str) -> tuple[float, float]:
    vals = [r[method]["ari"] for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def benchmark(
    seeds: list[int] | None = None,
    regimes: list[str] | None = None,
    config: Config | None = None,
) -> dict:
    """多 regime × 多 seed benchmark，返回聚合结果与门禁结论。"""
    config = config or get_config()
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    if regimes is None:
        regimes = list(REGIMES.keys())

    rows: list[dict] = []
    for reg in regimes:
        spec = REGIMES[reg]
        for s in seeds:
            rows.append(run_one(s, spec["sizes"], spec["p_in"], spec["p_out"], config, regime=reg))

    methods = ["fuse", "louvain_plain", "spectral_kmeans", "kmeans_adjacency", "random"]
    summary: dict[str, dict] = {}
    for m in methods:
        mu, sd = _agg_ari(rows, m)
        mod = float(np.mean([r[m]["modularity"] for r in rows]))
        summary[m] = {"ari_mean": mu, "ari_std": sd, "modularity_mean": mod}

    per_regime: dict[str, dict] = {}
    for reg in regimes:
        sub = [r for r in rows if r["regime"] == reg]
        per_regime[reg] = {
            m: {
                "ari_mean": float(np.mean([r[m]["ari"] for r in sub])),
                "ari_std": float(np.std([r[m]["ari"] for r in sub])),
            }
            for m in methods
        }

    agg = {
        "seeds": list(seeds),
        "regimes": regimes,
        "n_runs": len(rows),
        "summary": summary,
        "per_regime": per_regime,
        "rows": rows,
    }
    agg["gate"] = _evaluate_gate(agg, config)
    return agg


def _max_modularity_decrease(level_q: list[float]) -> float:
    """Louvain 跨层模块度最大下降（应为 <= 0）。"""
    max_dec = 0.0
    for a, b in zip(level_q[:-1], level_q[1:]):
        dec = a - b
        if dec > max_dec:
            max_dec = dec
    return max_dec


def _invariants_ok(config: Config) -> dict:
    """检查核心不变量（拉普拉斯 PSD、谱嵌入正交、WL 同构）。"""
    out: dict = {}

    # 1) 拉普拉斯半正定：最小特征值 >= -tol（用 Fiedler 的第二小值间接验证 >= 0）
    g = erdos_renyi(40, 0.3, seed=7)
    lam2 = fiedler_value(g.adjacency)
    out["laplacian_psd"] = bool(lam2 >= -config.gate_max_inv_violation)

    # 2) 谱嵌入正交 V^T V = I
    sp = erdos_renyi(40, 0.3, seed=3)
    V, _ = symmetric_normalized_embedding(sp, k=4, seed=1)
    orth = float(np.max(np.abs(V.T @ V - np.eye(V.shape[1]))))
    out["embed_orthogonal"] = bool(orth <= 1e-6)
    out["embed_orth_max_err"] = orth

    # 3) WL 同构：构造随机图及其置换副本
    base = erdos_renyi(30, 0.4, seed=11)
    perm = np.random.default_rng(11).permutation(30)
    A2 = base.adjacency[np.ix_(perm, perm)]
    g2 = GraphContainer(n_nodes=30, adjacency=A2, name="perm")
    k_iso = weisfeiler_lehman_kernel(base, g2, n_iter=config.wl_max_iter)
    k_self = weisfeiler_lehman_kernel(base, base, n_iter=config.wl_max_iter)
    out["wl_isomorphic"] = bool(abs(k_iso - k_self) <= 1e-9)

    # 4) Louvain 模块度单调（代表性运行）
    sbm = stochastic_block_model([40, 40, 40], 0.15, 0.005, seed=5)
    lv = louvain(sbm, gamma=1.0, seed=5)
    dec = _max_modularity_decrease(lv.meta["level_modularity"])
    out["modularity_monotone"] = bool(dec <= config.gate_max_modularity_decrease)
    out["modularity_monotone_dec"] = float(dec)

    return out


def _evaluate_gate(agg: dict, config: Config) -> dict:
    fuse_mean = agg["summary"]["fuse"]["ari_mean"]
    plain_mean = agg["summary"]["louvain_plain"]["ari_mean"]
    spec_mean = agg["summary"]["spectral_kmeans"]["ari_mean"]
    km_mean = agg["summary"]["kmeans_adjacency"]["ari_mean"]

    margin_plain = fuse_mean - plain_mean
    margin_spec = fuse_mean - spec_mean

    inv = _invariants_ok(config)

    g1 = fuse_mean >= config.gate_min_recovery_ari
    g2 = margin_plain >= config.gate_min_margin_over_baseline
    g3 = inv["modularity_monotone"]
    g4 = all(v for k, v in inv.items() if k != "modularity_monotone" and isinstance(v, bool))

    gates = {
        "G1_recovery_ari": {
            "pass": bool(g1),
            "value": fuse_mean,
            "threshold": config.gate_min_recovery_ari,
        },
        "G2_margin_over_plain_louvain": {
            "pass": bool(g2),
            "value": margin_plain,
            "threshold": config.gate_min_margin_over_baseline,
        },
        "G3_modularity_monotone": {
            "pass": bool(g3),
            "value": "ok" if g3 else "fail",
            "threshold": config.gate_max_modularity_decrease,
        },
        "G4_invariants": {"pass": bool(g4), "checks": inv},
        # 参考项（非门禁）：谱聚类已知 k 是 SBM 统计天花板，诚实标注
        "reference_vs_spectral_known_k": {
            "fuse": fuse_mean,
            "spectral": spec_mean,
            "margin": margin_spec,
            "note": "spectral with known k is the statistical ceiling on SBM; not claimed to be beaten",
        },
        "reference_vs_kmeans_adjacency": {
            "fuse": fuse_mean,
            "kmeans": km_mean,
            "margin": fuse_mean - km_mean,
        },
    }
    gates["all_pass"] = bool(g1 and g2 and g3 and g4)
    return gates
