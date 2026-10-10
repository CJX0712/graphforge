"""图构造、图拉普拉斯与可证不变量。

作者：晨星

核心不变量（手写实现硬金标准）：
- 拉普拉斯 L = D - A 对称半正定 → 特征值 >= -tol
- 随机游走归一化拉普拉斯 L_rw = I - D^{-1}A 行和为 0
- 对称归一化拉普拉斯 L_sym = I - D^{-1/2} A D^{-1/2} 对称、特征值 in [0,2]
- Fiedler 值（第二小特征值）> 0 当且仅当图连通
"""

from __future__ import annotations

import numpy as np

from ..core.errors import (
    E100_INVALID_GRAPH,
    E101_NOT_SYMMETRIC,
    E102_NEGATIVE_WEIGHT,
)
from ..core.interfaces import GraphContainer
from ..core.seed import new_rng


def _validate(A: np.ndarray) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise E100_INVALID_GRAPH("邻接矩阵必须为方阵")
    if not np.allclose(A, A.T, atol=1e-12):
        raise E101_NOT_SYMMETRIC("邻接矩阵必须对称")
    if np.any(A < -1e-15):
        raise E102_NEGATIVE_WEIGHT("邻接矩阵不可含负权")


def erdos_renyi(n: int, p: float, seed: int = 0, directed: bool = False) -> GraphContainer:
    """Erdős–Rényi G(n, p) 随机图生成器。

    确定性：仅依赖 seed。返回无自环简单图（对角强制 0）。
    """
    rng = new_rng(seed)
    mat = rng.random((n, n)) < p
    A = mat.astype(np.float64)
    if not directed:
        A = np.triu(A, 1)
        A = A + A.T
    np.fill_diagonal(A, 0.0)
    return GraphContainer(n_nodes=n, adjacency=A, name=f"ER(n={n},p={p})")


def stochastic_block_model(
    sizes: list[int],
    p_in: float,
    p_out: float,
    seed: int = 0,
) -> GraphContainer:
    """随机块模型（planted partition）。

    每个块内部连边概率 p_in，块间 p_out。返回图 + 真实社区标签。

    Returns
    -------
    GraphContainer
        labels 字段为真实社区 id（0..K-1）。
    """
    rng = new_rng(seed)
    sizes = list(sizes)
    n = int(sum(sizes))
    k = len(sizes)
    block = np.zeros(n, dtype=int)
    idx = 0
    for b, sz in enumerate(sizes):
        block[idx : idx + sz] = b
        idx += sz

    A = np.zeros((n, n), dtype=np.float64)
    # 逐块生成，保证对称
    for a in range(k):
        for b in range(a, k):
            if a == b:
                prob = p_in
            else:
                prob = p_out
            i0, i1 = int(sum(sizes[:a])), int(sum(sizes[:a]) + sizes[a])
            j0, j1 = int(sum(sizes[:b])), int(sum(sizes[:b]) + sizes[b])
            sub = rng.random((sizes[a], sizes[b])) < prob
            if a == b:
                # 块内：取严格上三角 + 其转置 ⇒ 对称且无自环（对角为 0）
                tri = np.triu(sub, 1).astype(np.float64)
                A[i0:i1, i0:i1] = tri + tri.T
            else:
                A[i0:i1, j0:j1] = sub.astype(np.float64)
                A[j0:j1, i0:i1] = sub.T.astype(np.float64)
    np.fill_diagonal(A, 0.0)
    return GraphContainer(n_nodes=n, adjacency=A, labels=block, name="SBM")


def from_edge_list(
    edges: list[tuple[int, int, float]],
    n_nodes: int | None = None,
    labels: np.ndarray | None = None,
    name: str = "edge_list",
) -> GraphContainer:
    """从边列表构造图 (u, v, weight)。无向、自动对称。"""
    if n_nodes is None:
        n_nodes = 0
        for u, v, _w in edges:
            n_nodes = max(n_nodes, u + 1, v + 1)
    A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for u, v, w in edges:
        A[u, v] += w
        A[v, u] += w
    np.fill_diagonal(A, 0.0)
    return GraphContainer(n_nodes=int(n_nodes), adjacency=A, labels=labels, name=name)


_KARATE_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
]


def karate_club() -> GraphContainer:
    """Zachary (1977) 空手道俱乐部网络（34 节点，78 边）。

    无自环、确定性、可复现。常用于社区发现基准。
    """
    n = 34
    edges = [(u, v, 1.0) for u, v in _KARATE_EDGES]
    return from_edge_list(edges, n_nodes=n, name="karate_club")


def laplacian(A: np.ndarray, kind: str = "unnormalized") -> np.ndarray:
    """计算图拉普拉斯。

    Parameters
    ----------
    A : np.ndarray
        对称邻接矩阵。
    kind : {"unnormalized", "symmetric_normalized", "random_walk"}
        - unnormalized: L = D - A
        - symmetric_normalized: L_sym = I - D^{-1/2} A D^{-1/2}
        - random_walk: L_rw = I - D^{-1} A
    """
    _validate(A)
    d = A.sum(axis=1)
    if kind == "unnormalized":
        D = np.diag(d)
        return D - A
    # 避免除零（孤立节点）
    dinv = np.zeros_like(d)
    nz = d > 0
    dinv[nz] = 1.0 / d[nz]
    if kind == "random_walk":
        return np.eye(A.shape[0]) - (A * dinv[:, None])
    # symmetric_normalized
    dsqrt = np.sqrt(dinv)
    L = np.eye(A.shape[0]) - (dsqrt[:, None] * A * dsqrt[None, :])
    return L


def fiedler_value(A: np.ndarray) -> float:
    """Fiedler 值 = 无归一化拉普拉斯第二小特征值。

    > 0 当且仅当图连通（唯一连通分量时最小特征值为 0）。
    """
    from ._eig import symeig_smallest

    L = laplacian(A, "unnormalized")
    _V, lam = symeig_smallest(L, 2, iters=2000, seed=0)
    return float(lam[1])


def is_connected(A: np.ndarray) -> bool:
    """基于 Fiedler 值的连通性判定。"""
    return fiedler_value(A) > 1e-9


def connected_components(A: np.ndarray) -> tuple[int, np.ndarray]:
    """BFS 连通分量（纯 NumPy）。返回 (n_components, labels)。"""
    n = A.shape[0]
    labels = -np.ones(n, dtype=int)
    comp = 0
    for s in range(n):
        if labels[s] != -1:
            continue
        # BFS
        stack = [s]
        labels[s] = comp
        while stack:
            u = stack.pop()
            nbrs = np.where(A[u] > 0)[0]
            for v in nbrs:
                if labels[v] == -1:
                    labels[v] = comp
                    stack.append(v)
        comp += 1
    return comp, labels
