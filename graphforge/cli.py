"""graphForge 命令行入口。

作者：晨星

用法：
    python -m graphforge.cli benchmark [--seeds 0,1,2,3,4] [--out benchmark.json]
    python -m graphforge.cli karate
    python -m graphforge.cli wl --n 25 --seed 2
    python -m graphforge.cli pagerank --n 30 --p 0.2 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from .core.config import get_config
from .core.interfaces import GraphContainer
from .core.seed import set_all
from .graph.build import erdos_renyi, karate_club
from .graph.centrality import pagerank
from .graph.community import community_fuse
from .graph.kernels import weisfeiler_lehman_kernel
from .pipeline.pipeline import benchmark


def _cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = get_config()
    seeds = [int(s) for s in args.seeds.split(",")]
    set_all(cfg.default_seed)
    out = benchmark(seeds=seeds, config=cfg)
    s = out["summary"]
    print("=== graphForge benchmark（多 regime SBM 社区发现）===")
    for m in ["fuse", "louvain_plain", "spectral_kmeans", "kmeans_adjacency", "random"]:
        v = s[m]
        print(f"  {m:<16} ARI {v['ari_mean']:.4f} ± {v['ari_std']:.4f}   Q {v['modularity_mean']:.4f}")
    print()
    for reg, d in out["per_regime"].items():
        print(f"  {reg:<18} fuse ARI {d['fuse']['ari_mean']:.4f} | plain {d['louvain_plain']['ari_mean']:.4f}")
    print()
    print(json.dumps(out["gate"], ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"written -> {args.out}")
    return 0 if out["gate"]["all_pass"] else 1


def _cmd_karate(args: argparse.Namespace) -> int:
    kc = karate_club()
    res = community_fuse(kc, seed=args.seed)
    print(f"karate_club: n={kc.n_nodes} communities={res.n_communities} modularity={res.modularity:.4f}")
    return 0


def _cmd_wl(args: argparse.Namespace) -> int:
    g1 = erdos_renyi(args.n, 0.5, seed=args.seed)
    perm = np.random.default_rng(args.seed).permutation(args.n)
    A2 = g1.adjacency[np.ix_(perm, perm)]
    g2 = GraphContainer(n_nodes=args.n, adjacency=A2, name="perm")
    k_iso = weisfeiler_lehman_kernel(g1, g2, n_iter=args.n_iter)
    k_self = weisfeiler_lehman_kernel(g1, g1, n_iter=args.n_iter)
    print(f"WL iso kernel={k_iso:.4f} self kernel={k_self:.4f} (应相等)")
    return 0


def _cmd_pagerank(args: argparse.Namespace) -> int:
    g = erdos_renyi(args.n, args.p, seed=args.seed)
    pr = pagerank(g, alpha=args.alpha)
    top = np.argsort(pr)[::-1][: min(5, args.n)].tolist()
    print(f"PageRank sum={pr.sum():.6f} top5 nodes={top}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphforge", description="graphForge CLI · 作者 晨星")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="多 regime SBM 社区发现 benchmark + 门禁")
    b.add_argument("--seeds", default="0,1,2,3,4")
    b.add_argument("--out", default="")
    b.set_defaults(func=_cmd_benchmark)

    k = sub.add_parser("karate", help="空手道俱乐部社区发现")
    k.add_argument("--seed", type=int, default=0)
    k.set_defaults(func=_cmd_karate)

    w = sub.add_parser("wl", help="Weisfeiler-Lehman 同构核演示")
    w.add_argument("--n", type=int, default=25)
    w.add_argument("--seed", type=int, default=2)
    w.add_argument("--n-iter", dest="n_iter", type=int, default=5)
    w.set_defaults(func=_cmd_wl)

    p = sub.add_parser("pagerank", help="PageRank 中心性演示")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--p", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.85)
    p.set_defaults(func=_cmd_pagerank)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
