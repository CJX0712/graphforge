"""谱聚类测试：嵌入正交性、特征值范围、聚类质量。

作者：晨星
"""

import numpy as np

from graphforge.graph.build import erdos_renyi, from_edge_list, stochastic_block_model
from graphforge.graph.spectral import (
    spectral_clustering,
    symmetric_normalized_embedding,
)


def test_embedding_orthonormal():
    """V^T V = I（Gram-Schmidt 保证），误差 ~1e-16。"""
    g = erdos_renyi(40, 0.3, seed=1)
    V, lam = symmetric_normalized_embedding(g, k=4, seed=1)
    err = np.max(np.abs(V.T @ V - np.eye(4)))
    assert err < 1e-8, f"正交误差过大: {err}"


def test_eigenvalues_in_range():
    """对称归一化拉普拉斯特征值 ∈ [0, 2]。"""
    g = stochastic_block_model([30, 30, 30], 0.2, 0.01, seed=3)
    V, lam = symmetric_normalized_embedding(g, k=4, seed=3)
    assert np.all(lam >= -1e-8)
    assert np.all(lam <= 2.0 + 1e-8)
    # 升序
    assert np.all(np.diff(lam) >= -1e-8)


def test_spectral_recovers_two_cliques():
    """两个分离团：谱聚类应完美恢复（已知真值）。"""
    from graphforge.eval.metrics import ari

    k, m = 2, 15
    lab = np.repeat(np.arange(k), m)
    n = k * m
    A = np.zeros((n, n))
    for c in range(k):
        idx = np.where(lab == c)[0]
        A[np.ix_(idx, idx)] = 1.0
    np.fill_diagonal(A, 0.0)
    from graphforge.core.interfaces import GraphContainer

    g = GraphContainer(n_nodes=n, adjacency=A, labels=lab, name="two_cliques")
    res = spectral_clustering(g, k=2, seed=0)
    assert ari(lab, res.labels) > 0.99


def test_spectral_recovers_sbm():
    from graphforge.eval.metrics import ari

    g = stochastic_block_model([50, 50, 50, 50], 0.15, 0.005, seed=0)
    res = spectral_clustering(g, k=4, seed=0)
    assert ari(g.labels, res.labels) > 0.90


def test_trivial_eigenvector_is_constant_for_connected_graph():
    """连通图归一化拉普拉斯最小特征值为 0，对应特征向量（缩放后）为常数方向。"""
    n = 24
    edges = [(i, (i + 1) % n, 1.0) for i in range(n)]
    g = from_edge_list(edges, n_nodes=n)
    V, lam = symmetric_normalized_embedding(g, k=2, seed=0)
    assert lam[0] < 1e-6
    v0 = V[:, 0]
    # 常数方向：归一化后各分量绝对值相等
    absv = np.abs(v0)
    assert np.allclose(absv, absv.mean(), atol=1e-6)
