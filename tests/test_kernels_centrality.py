"""图核（WL）与中心性（PageRank）测试。

作者：晨星
"""

import numpy as np

from graphforge.core.interfaces import GraphContainer
from graphforge.graph.build import erdos_renyi, from_edge_list
from graphforge.graph.centrality import pagerank
from graphforge.graph.kernels import weisfeiler_lehman_kernel, wl_features


def _permute(g: GraphContainer, seed: int) -> GraphContainer:
    perm = np.random.default_rng(seed).permutation(g.n_nodes)
    A2 = g.adjacency[np.ix_(perm, perm)]
    return GraphContainer(n_nodes=g.n_nodes, adjacency=A2, name="permuted")


def test_wl_isomorphic_graphs_equal_kernel():
    """同构图（节点置换）核值 == 自核。"""
    for n, p in [(20, 0.4), (30, 0.3)]:
        g = erdos_renyi(n, p, seed=1)
        g2 = _permute(g, seed=1)
        k_iso = weisfeiler_lehman_kernel(g, g2, n_iter=5)
        k_self = weisfeiler_lehman_kernel(g, g, n_iter=5)
        assert abs(k_iso - k_self) <= 1e-9, f"iso={k_iso} self={k_self}"


def test_wl_non_isomorphic_strictly_smaller():
    """结构不同（环 vs 路径）核值严格小于自核。"""
    n = 6
    cycle = from_edge_list([(i, (i + 1) % n, 1.0) for i in range(n)], n_nodes=n)
    path = from_edge_list([(i, i + 1, 1.0) for i in range(n - 1)], n_nodes=n)
    k_cross = weisfeiler_lehman_kernel(cycle, path, n_iter=3)
    k_self_c = weisfeiler_lehman_kernel(cycle, cycle, n_iter=3)
    k_self_p = weisfeiler_lehman_kernel(path, path, n_iter=3)
    assert k_cross < k_self_c
    assert k_cross < k_self_p


def test_wl_features_deterministic_and_nonneg():
    g = erdos_renyi(20, 0.4, seed=5)
    f1 = wl_features(g, n_iter=3)
    f2 = wl_features(g, n_iter=3)
    assert np.array_equal(f1, f2)
    assert np.all(f1 >= 0)


def test_wl_kernel_symmetric():
    g1 = erdos_renyi(18, 0.4, seed=7)
    g2 = erdos_renyi(22, 0.35, seed=8)
    k12 = weisfeiler_lehman_kernel(g1, g2, n_iter=3)
    k21 = weisfeiler_lehman_kernel(g2, g1, n_iter=3)
    assert abs(k12 - k21) <= 1e-9


def test_pagerank_sums_to_one():
    for seed in range(3):
        g = erdos_renyi(30, 0.25, seed=seed)
        pr = pagerank(g, alpha=0.85)
        assert np.isclose(pr.sum(), 1.0, atol=1e-9)
        assert np.all(pr >= -1e-12)


def test_pagerank_uniform_on_complete_graph():
    """完全图（所有节点对称）PageRank 应均匀。"""
    n = 12
    edges = [(i, j, 1.0) for i in range(n) for j in range(i + 1, n)]
    g = from_edge_list(edges, n_nodes=n)
    pr = pagerank(g, alpha=0.85)
    assert np.allclose(pr, 1.0 / n, atol=1e-9)


def test_pagerank_star_center_highest():
    """星形图中心节点得分最高。"""
    n = 10
    edges = [(0, i, 1.0) for i in range(1, n)]
    g = from_edge_list(edges, n_nodes=n)
    pr = pagerank(g, alpha=0.85)
    assert int(np.argmax(pr)) == 0
