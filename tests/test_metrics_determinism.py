"""指标正确性与确定性测试。

作者：晨星
"""

import numpy as np

from graphforge.eval.metrics import ari, nmi, variation_of_information
from graphforge.graph.build import stochastic_block_model
from graphforge.graph.community import community_fuse, louvain
from graphforge.graph.spectral import spectral_clustering


def test_ari_perfect_and_random():
    y = np.repeat([0, 1, 2], 10)
    assert np.isclose(ari(y, y), 1.0)
    # 随机划分 ARI ≈ 0
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 3, size=30)
    assert abs(ari(y, pred)) < 0.15


def test_ari_known_value():
    """经典例子：ARI 对标签置换不变。"""
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([1, 1, 0, 0, 2, 2])  # 仅置换
    assert np.isclose(ari(y, p), 1.0)


def test_nmi_and_vi():
    y = np.repeat([0, 1], 20)
    assert np.isclose(nmi(y, y), 1.0)
    assert np.isclose(variation_of_information(y, y), 0.0, atol=1e-12)


def test_determinism_same_seed_identical():
    """同 seed 两次运行：划分与 ARI 逐位一致。"""
    for seed in [0, 1]:
        g1 = stochastic_block_model([50, 50, 50], 0.18, 0.008, seed=seed)
        g2 = stochastic_block_model([50, 50, 50], 0.18, 0.008, seed=seed)
        assert np.array_equal(g1.adjacency, g2.adjacency)

        f1 = community_fuse(g1, k=3, seed=seed)
        f2 = community_fuse(g2, k=3, seed=seed)
        assert np.array_equal(f1.labels, f2.labels)
        assert f1.modularity == f2.modularity

        s1 = spectral_clustering(g1, k=3, seed=seed)
        s2 = spectral_clustering(g2, k=3, seed=seed)
        assert np.array_equal(s1.labels, s2.labels)


def test_louvain_deterministic():
    g = stochastic_block_model([60, 60], 0.15, 0.005, seed=3)
    a = louvain(g, seed=3)
    b = louvain(g, seed=3)
    assert np.array_equal(a.labels, b.labels)
