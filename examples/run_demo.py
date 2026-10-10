"""graphForge 端到端演示：benchmark + 确定性双跑 + 门禁核验 + 消融 + 失败案例。

作者：晨星

产出：benchmark.json（核心指标，二次运行逐位一致）。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# 允许脚本直接运行：把仓库根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from graphforge.core.config import get_config  # noqa: E402
from graphforge.core.interfaces import GraphContainer  # noqa: E402
from graphforge.core.seed import set_all  # noqa: E402
from graphforge.eval.metrics import ari  # noqa: E402
from graphforge.graph.build import (  # noqa: E402
    erdos_renyi,
    karate_club,
    stochastic_block_model,
)
from graphforge.graph.centrality import pagerank  # noqa: E402
from graphforge.graph.community import community_fuse, louvain  # noqa: E402
from graphforge.graph.kernels import weisfeiler_lehman_kernel  # noqa: E402
from graphforge.pipeline.pipeline import REGIMES, benchmark  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def _r(x, n: int = 4) -> str:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return f"{x:.{n}f}"
    return str(x)


def run_demo(out_path: str = "benchmark.json") -> dict:
    cfg = get_config()
    set_all(cfg.default_seed)

    print("=" * 74)
    print("graphForge · 纯 NumPy 谱图学习与社区发现  ·  作者 晨星")
    print("=" * 74)

    # ---------- 1. 多 regime benchmark ----------
    print("\n[1] SBM 社区发现 benchmark（3 regime × 5 seeds = 15 runs）")
    print(f"    {'方法':<20}{'ARI mean':>10}{'ARI std':>10}{'Mod mean':>10}  说明")
    agg = benchmark(config=cfg)
    s = agg["summary"]
    notes = {
        "fuse": "旗舰 CommunityFuse（给 k）",
        "spectral_kmeans": "天花板参考（给 k）",
        "louvain_plain": "领域标准基线（不给 k）",
        "kmeans_adjacency": "邻接行 k-means（给 k）",
        "random": "随机划分",
    }
    for m in ["fuse", "louvain_plain", "spectral_kmeans", "kmeans_adjacency", "random"]:
        v = s[m]
        print(f"    {m:<20}{_r(v['ari_mean']):>10}{_r(v['ari_std']):>10}{_r(v['modularity_mean']):>10}  {notes[m]}")

    print("\n    分 regime ARI mean:")
    for reg, d in agg["per_regime"].items():
        spec = REGIMES[reg]
        print(
            f"      {reg:<18} sizes={str(spec['sizes']):<18} "
            f"p_in={spec['p_in']} p_out={spec['p_out']}  "
            f"fuse={_r(d['fuse']['ari_mean'])}  "
            f"louvain={_r(d['louvain_plain']['ari_mean'])}  "
            f"spectral={_r(d['spectral_kmeans']['ari_mean'])}"
        )

    # ---------- 2. 门禁 ----------
    print("\n[2] 门禁核验")
    gate = agg["gate"]
    for name, g in gate.items():
        if name == "all_pass" or name.startswith("reference_"):
            continue
        status = "✅" if g["pass"] else "❌"
        if name == "G4_invariants":
            print(f"    {status} {name}: {g['checks']}")
        else:
            thr = g.get("threshold")
            if thr is None:
                print(f"    {status} {name}: value={g['value']}")
            else:
                print(f"    {status} {name}: value={_r(g['value'])}  threshold={_r(thr)}")
    print(f"    {'✅' if gate['all_pass'] else '❌'} 全部门禁通过: {gate['all_pass']}")

    ref = gate["reference_vs_spectral_known_k"]
    print(
        f"    ℹ 参考项（非门禁）：谱聚类已知 k 是 SBM 统计天花板 "
        f"fuse={_r(ref['fuse'])} spectral={_r(ref['spectral'])} margin={_r(ref['margin'])}"
    )
    print("       → 诚实声明：本系统不声称超越该天花板，贡献点是修复模块度分辨率极限。")

    # ---------- 3. 确定性双跑 ----------
    print("\n[3] 确定性双跑校验（同 seed 两次 benchmark 核心指标逐位一致）")
    agg2 = benchmark(config=cfg)
    det_ok = (
        agg["summary"]["fuse"]["ari_mean"] == agg2["summary"]["fuse"]["ari_mean"]
        and agg["summary"]["louvain_plain"]["ari_mean"] == agg2["summary"]["louvain_plain"]["ari_mean"]
        and agg["summary"]["spectral_kmeans"]["ari_mean"] == agg2["summary"]["spectral_kmeans"]["ari_mean"]
    )
    print(f"    {'✅' if det_ok else '❌'} 核心指标逐位一致: {det_ok}")

    # ---------- 4. 消融 ----------
    print("\n[4] 消融（k 归并 + 精炼 组件开关，unbalanced regime，5 seeds）")
    abl = {}
    spec = REGIMES["unbalanced"]
    with_c, without_c = [], []
    for sd in range(5):
        g = stochastic_block_model(spec["sizes"], spec["p_in"], spec["p_out"], seed=sd)
        true = g.labels
        f = community_fuse(g, k=4, seed=sd)
        p = louvain(g, seed=sd)
        with_c.append(float(ari(true, f.labels)))
        without_c.append(float(ari(true, p.labels)))
    abl["unbalanced"] = {
        "with_consolidation_and_refine": float(np.mean(with_c)),
        "without_louvain_plain": float(np.mean(without_c)),
        "delta": float(np.mean(with_c) - np.mean(without_c)),
    }
    print(
        f"    含组件 {_r(abl['unbalanced']['with_consolidation_and_refine'])} "
        f"vs 关闭（普通 Louvain）{_r(abl['unbalanced']['without_louvain_plain'])} "
        f"→ Δ={_r(abl['unbalanced']['delta'])}"
    )

    # ---------- 5. 失败案例 ----------
    print("\n[5] 失败案例（全部从运行结果派生，含归因）")
    worst = sorted(agg["rows"], key=lambda r: r["fuse"]["ari"])[:3]
    failures = []
    for r in worst:
        cause = []
        if r["fuse"]["n_communities"] != r["k"]:
            cause.append(f"社区数 {r['fuse']['n_communities']} != k={r['k']}")
        if r["regime"] == "balanced_moderate":
            cause.append("信号较弱(p_in=0.12/p_out=0.010)，接近可检测阈值")
        if r["regime"] == "unbalanced" and r["fuse_strategy"] == "consolidate_modularity":
            cause.append("贪心归并策略在不平衡图上易误并")
        failures.append(
            {
                "regime": r["regime"],
                "seed": r["seed"],
                "fuse_ari": r["fuse"]["ari"],
                "spectral_ari": r["spectral_kmeans"]["ari"],
                "louvain_ari": r["louvain_plain"]["ari"],
                "strategy": r["fuse_strategy"],
                "n_communities": r["fuse"]["n_communities"],
                "cause": "; ".join(cause) or "结构噪声导致边界节点误分",
            }
        )
        print(
            f"    regime={r['regime']} seed={r['seed']} fuse_ARI={_r(r['fuse']['ari'])} "
            f"(spectral {_r(r['spectral_kmeans']['ari'])}) 归因: {failures[-1]['cause']}"
        )

    # ---------- 6. 展示性组件 ----------
    print("\n[6] 展示性组件")
    kc = karate_club()
    kc_fuse = community_fuse(kc, k=2, seed=0)
    print(f"    空手道俱乐部(34节点): CommunityFuse(k=2) 模块度={_r(kc_fuse.modularity)}")

    g1 = erdos_renyi(25, 0.5, seed=2)
    perm = np.random.default_rng(2).permutation(25)
    g2p = GraphContainer(n_nodes=25, adjacency=g1.adjacency[np.ix_(perm, perm)], name="perm")
    g3 = erdos_renyi(25, 0.45, seed=9)  # 非同构但结构相近（非零但更小的核值）
    k_iso = weisfeiler_lehman_kernel(g1, g2p, n_iter=5)
    k_self = weisfeiler_lehman_kernel(g1, g1, n_iter=5)
    k_diff = weisfeiler_lehman_kernel(g1, g3, n_iter=5)
    print(f"    WL 核: iso={_r(k_iso)} self={_r(k_self)}（应相等） | 非同构={_r(k_diff)}（应更小）")

    pr = pagerank(kc, alpha=0.85)
    top3 = np.argsort(pr)[::-1][:3].tolist()
    print(f"    PageRank 空手道 Top3 节点: {top3}  得分和={_r(pr.sum())}")

    # ---------- 7. 落盘 ----------
    out = {
        "system": "graphForge",
        "version": "0.1.0",
        "author": "晨星",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.as_dict(),
        "benchmark": {
            "n_runs": agg["n_runs"],
            "seeds": agg["seeds"],
            "regimes": agg["regimes"],
            "summary": agg["summary"],
            "per_regime": agg["per_regime"],
        },
        "gate": gate,
        "determinism_bit_identical": det_ok,
        "ablation": abl,
        "failure_cases": failures,
        "karate_modularity": float(kc_fuse.modularity),
        "wl_iso_kernel": k_iso,
        "wl_self_kernel": k_self,
        "wl_diff_kernel": k_diff,
        "pagerank_sum": float(pr.sum()),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[7] 结果已落盘: {out_path}")
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmark.json"
    run_demo(path)
