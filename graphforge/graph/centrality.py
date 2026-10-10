"""图中心性：PageRank（幂迭代 + 悬挂节点处理，纯 NumPy）。

作者：晨星

数学内核（硬金标准）：
- 平稳分布 p = α·M·p + (1-α)/n，M 列随机（含悬挂节点均匀重分配）。
- 不变量：Σ p_i = 1（±tol）；完全图 p 均匀；星形图中心节点得分最高。
"""

from __future__ import annotations

import numpy as np

from ..core.interfaces import GraphContainer


def pagerank(
    graph: GraphContainer,
    alpha: float = 0.85,
    iters: int = 500,
    tol: float = 1e-12,
) -> np.ndarray:
    """PageRank 中心性。返回各节点得分（和为 1）。"""
    A = graph.adjacency
    n = A.shape[0]
    if n == 0:
        return np.zeros(0)
    d = A.sum(axis=1)
    M = np.zeros((n, n))
    nz = d > 0
    if nz.any():
        M[:, nz] = A[:, nz] / d[nz][None, :]
    dang = ~nz
    if dang.any():
        M[:, dang] = 1.0 / n
    p = np.full(n, 1.0 / n)
    for _ in range(iters):
        p_new = alpha * (M @ p) + (1.0 - alpha) / n
        if np.linalg.norm(p_new - p) < tol:
            p = p_new
            break
        p = p_new
    return p
