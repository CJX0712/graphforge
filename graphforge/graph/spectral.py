"""纯 NumPy 谱聚类：对称归一化拉普拉斯最小特征向量（幂迭代 + 收缩）。

作者：晨星

数学内核（硬金标准）：
- M = I - D^{-1/2} A D^{-1/2} 对称，特征值 ∈ [0, 2]。
- 通过 (cI - M) 幂迭代 + 收缩，无 scipy 求 k 个最小特征向量。
- 不变量：返回特征向量正交（V^T V = I ± tol），特征值 ∈ [0, 2 ± tol]。
- 嵌入后行归一化 + 确定性 k-means（k-means++ 种子固定）得到聚类。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E100_INVALID_GRAPH
from ..core.interfaces import GraphContainer, PartitionResult
from ..core.seed import new_rng


def _symeig_smallest(M: np.ndarray, k: int, iters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """求对称矩阵 M 最小 k 个特征值对应的特征向量（委托共享求解器）。"""
    from ._eig import symeig_smallest

    return symeig_smallest(M, k, iters=iters, seed=seed)


def symmetric_normalized_embedding(graph: GraphContainer, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """返回对称归一化拉普拉斯的最小 k 个特征向量（嵌入）与对应特征值。

    不变量：V^T V ≈ I，特征值 ∈ [0, 2]。
    """
    from ..core.config import get_config

    cfg = get_config()
    A = graph.adjacency
    if A.shape[0] < k:
        raise E100_INVALID_GRAPH("节点数小于聚类数 k")
    d = A.sum(axis=1)
    dinv = np.zeros_like(d)
    nz = d > 0
    dinv[nz] = 1.0 / d[nz]
    dsqrt = np.sqrt(dinv)
    M = np.eye(A.shape[0]) - (dsqrt[:, None] * A * dsqrt[None, :])
    V, lam = _symeig_smallest(M, k, cfg.spectral_power_iters, seed)
    return V, lam


def _kmeans(X: np.ndarray, k: int, iters: int, tol: float, seed: int) -> np.ndarray:
    """纯 NumPy k-means++（确定性种子）。返回每个样本簇标签。"""
    n, dim = X.shape
    rng = new_rng(seed)
    # k-means++ 初始化
    centers = np.empty((k, dim))
    # 第一个中心
    first = int(rng.integers(n))
    centers[0] = X[first]
    closest = np.sum((X - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        probs = closest.copy()
        total = probs.sum()
        if total <= 0:
            idx = int(rng.integers(n))
        else:
            probs = probs / total
            idx = int(rng.choice(n, p=probs))
        centers[j] = X[idx]
        dist = np.sum((X - centers[j]) ** 2, axis=1)
        closest = np.minimum(closest, dist)
    # Lloyd 迭代
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
    return labels


def spectral_clustering(graph: GraphContainer, k: int, seed: int = 0, use_kmeans: bool = True) -> PartitionResult:
    """谱聚类（Ng–Jordan–Weiss 变体，纯 NumPy）。

    Parameters
    ----------
    graph : GraphContainer
    k : int
        聚类数。
    seed : int
        确定性种子。
    """
    from ..core.config import get_config

    cfg = get_config()
    V, lam = symmetric_normalized_embedding(graph, k, seed)
    # 行归一化（NJW）
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Un = V / norms
    labels = _kmeans(Un, k, cfg.spectral_kmeans_iters, cfg.spectral_kmeans_tol, seed + 3)
    # 连续化标签
    _, labels = np.unique(labels, return_inverse=True)
    from .community import modularity

    q = modularity(graph.adjacency, labels, 1.0)
    return PartitionResult(
        labels=labels,
        n_communities=int(labels.max()) + 1,
        modularity=q,
        score=q,
        meta={"eigenvalues": lam.tolist(), "method": "spectral_kmeans"},
    )
