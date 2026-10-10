"""图构造与拉普拉斯不变量测试。

作者：晨星
"""

import numpy as np
import pytest

from graphforge.core.errors import E101_NOT_SYMMETRIC
from graphforge.graph.build import (
    connected_components,
    erdos_renyi,
    fiedler_value,
    from_edge_list,
    is_connected,
    karate_club,
    laplacian,
    stochastic_block_model,
)


def test_er_symmetric_and_no_selfloop():
    g = erdos_renyi(30, 0.3, seed=0)
    assert np.allclose(g.adjacency, g.adjacency.T)
    assert np.all(np.diag(g.adjacency) == 0)


def test_sbm_symmetric_and_labels():
    g = stochastic_block_model([20, 20, 10], 0.2, 0.01, seed=1)
    assert np.allclose(g.adjacency, g.adjacency.T)
    assert np.all(np.diag(g.adjacency) == 0)
    assert g.labels is not None
    assert g.labels.shape[0] == 50
    # 真实社区数 3
    assert set(np.unique(g.labels)) == {0, 1, 2}


def test_sbm_planted_structure_stronger_inside():
    """块内密度应显著高于块间（DGP 正确性）。"""
    g = stochastic_block_model([60, 60, 60], 0.20, 0.01, seed=3)
    lab = g.labels
    mask_in = lab[:, None] == lab[None, :]
    np.fill_diagonal(mask_in, False)
    mask_out = ~mask_in
    np.fill_diagonal(mask_out, False)
    # 有序对计数：块内 A 之和 / 块内有序对数量
    dens_in = g.adjacency[mask_in].sum() / mask_in.sum()
    dens_out = g.adjacency[mask_out].sum() / mask_out.sum()
    assert dens_in > 10 * dens_out


def test_from_edge_list_and_karate():
    g = from_edge_list([(0, 1, 1.0), (1, 2, 2.0)], n_nodes=3)
    assert g.adjacency[0, 1] == 1.0
    assert g.adjacency[1, 2] == 2.0
    assert np.allclose(g.adjacency, g.adjacency.T)

    kc = karate_club()
    assert kc.n_nodes == 34
    assert int(kc.adjacency.sum() / 2) == 78


def test_laplacian_psd_and_row_sums():
    """L = D - A 对称半正定；L_rw 行和为 0。"""
    g = erdos_renyi(25, 0.4, seed=2)
    A = g.adjacency
    L = laplacian(A, "unnormalized")
    assert np.allclose(L, L.T)
    eig = np.linalg.eigvalsh(L)
    assert eig.min() >= -1e-10  # 半正定

    Lrw = laplacian(A, "random_walk")
    # L_rw = I - D^{-1}A，行和应为 0（对非孤立节点）
    d = A.sum(axis=1)
    rows = np.where(d > 0)[0]
    assert np.allclose(Lrw[rows].sum(axis=1), 0.0, atol=1e-10)


def test_fiedler_connectivity():
    """连通图 Fiedler > 0；两个不相连团块 Fiedler ≈ 0。"""
    # 连通：环图
    n = 20
    edges = [(i, (i + 1) % n, 1.0) for i in range(n)]
    ring = from_edge_list(edges, n_nodes=n)
    assert fiedler_value(ring.adjacency) > 1e-6
    assert is_connected(ring.adjacency)

    # 不连通：两个独立团
    g1 = stochastic_block_model([15, 15], 0.5, 0.0, seed=4)  # p_out=0 ⇒ 两个分离团
    assert fiedler_value(g1.adjacency) < 1e-6
    assert not is_connected(g1.adjacency)
    ncomp, labels = connected_components(g1.adjacency)
    assert ncomp == 2


def test_laplacian_rejects_asymmetric():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(E101_NOT_SYMMETRIC):
        laplacian(A)


def test_graph_container_rejects_bad_shape():
    from graphforge.core.interfaces import GraphContainer

    with pytest.raises(ValueError):
        GraphContainer(n_nodes=3, adjacency=np.zeros((2, 2)))
