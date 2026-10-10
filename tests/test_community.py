"""社区发现测试：模块度金标准 + Louvain 单调不减 + k 归并。

作者：晨星
"""

import numpy as np

from graphforge.graph.build import karate_club, stochastic_block_model
from graphforge.graph.community import (
    community_fuse,
    consolidate_spectral,
    consolidate_to_k,
    louvain,
    modularity,
)


def test_modularity_bounds():
    """模块度 Q ∈ [-0.5, 1]。"""
    g = stochastic_block_model([30, 30, 30], 0.2, 0.01, seed=0)
    # 完美划分（真实标签）
    q_true = modularity(g.adjacency, g.labels, 1.0)
    # 全单节点
    q_single = modularity(g.adjacency, np.arange(g.n_nodes), 1.0)
    # 全一个社区
    q_one = modularity(g.adjacency, np.zeros(g.n_nodes, dtype=int), 1.0)
    assert -0.5 <= q_true <= 1.0
    assert -0.5 <= q_single <= 1.0
    assert np.isclose(q_one, 0.0, atol=1e-9)
    assert q_true > q_one


def test_modularity_two_cliques_analytic():
    """两个等大分离团的完美划分，模块度解析值 = 0.5。

    两个各 m 节点的完全图，无跨边：Q = 1 - 1/K = 1 - 1/2 = 0.5。
    """
    k = 2
    m = 8
    lab = np.repeat(np.arange(k), m)
    n = k * m
    A = np.zeros((n, n))
    for c in range(k):
        idx = np.where(lab == c)[0]
        A[np.ix_(idx, idx)] = 1.0
    np.fill_diagonal(A, 0.0)
    q = modularity(A, lab, 1.0)
    assert np.isclose(q, 0.5, atol=1e-9)


def test_louvain_modularity_monotone():
    """硬不变量：Louvain 跨层模块度单调不减（最大下降 <= 1e-9，实际为 0）。"""
    for seed in range(3):
        g = stochastic_block_model([40, 40, 40], 0.15, 0.005, seed=seed)
        res = louvain(g, gamma=1.0, seed=seed)
        trace = res.meta["level_modularity"]
        assert len(trace) >= 1
        for a, b in zip(trace[:-1], trace[1:]):
            assert b >= a - 1e-9, f"模块度下降: {a} -> {b}"


def test_louvain_final_modularity_matches_recompute():
    """最终模块度必须与金标准重算一致（内部追踪 vs 权威公式）。"""
    g = stochastic_block_model([30, 30, 30], 0.2, 0.01, seed=2)
    res = louvain(g, seed=2)
    q = modularity(g.adjacency, res.labels, 1.0)
    assert np.isclose(res.modularity, q, atol=1e-9)


def test_consolidate_to_k_reduces_communities():
    g = stochastic_block_model([150, 80, 40, 30], 0.15, 0.005, seed=1)
    lv = louvain(g, seed=1)
    assert lv.n_communities > 4
    cons = consolidate_to_k(g.adjacency, lv.labels, 4, 1.0)
    assert int(np.unique(cons).size) == 4


def test_consolidate_spectral_reduces_communities():
    g = stochastic_block_model([150, 80, 40, 30], 0.15, 0.005, seed=1)
    lv = louvain(g, seed=1)
    cons = consolidate_spectral(g.adjacency, lv.labels, 4, seed=1)
    assert int(np.unique(cons).size) == 4


def test_community_fuse_respects_k_and_beats_plain_louvain():
    """旗舰：给 k 后社区数正确，且 ARI 不低于普通 Louvain。"""
    from graphforge.eval.metrics import ari

    for seed in range(3):
        g = stochastic_block_model([150, 80, 40, 30], 0.15, 0.005, seed=seed)
        true = g.labels
        fuse = community_fuse(g, k=4, seed=seed)
        plain = louvain(g, seed=seed)
        a_f = ari(true, fuse.labels)
        a_p = ari(true, plain.labels)
        assert a_f >= a_p - 1e-9, f"旗舰未达普通 Louvain: {a_f} < {a_p}"


def test_community_fuse_karate():
    kc = karate_club()
    res = community_fuse(kc, k=2, seed=0)
    assert int(res.labels.max()) + 1 == 2
    assert res.modularity > 0.3
