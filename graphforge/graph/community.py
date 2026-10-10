"""社区发现：Louvain 模块度优化 + 模块度金标准。

作者：晨星

数学内核（硬金标准）：
- 模块度 Q = (1/2m) Σ_c [ inc_c - γ · tot_c² / (2m) ]，inc_c 为块内有序边权和，tot_c 为块总度。
- Louvain 局部移动 + 聚合：每一步仅接受严格提升 Q 的移动 ⇒ 跨层模块度单调不减
  （不变量：max_decrease_across_levels <= 1e-9，实际应为 0）。
- 模块度有界：Q ∈ [-0.5, 1]（对无向图）。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E100_INVALID_GRAPH, E102_NEGATIVE_WEIGHT
from ..core.interfaces import GraphContainer, PartitionResult
from ..core.seed import new_rng


def modularity(A: np.ndarray, comm: np.ndarray, gamma: float = 1.0) -> float:
    """金标准模块度（权威重算，用于门禁交叉验证）。

    Parameters
    ----------
    A : np.ndarray
        对称邻接矩阵（对角为 0）。
    comm : np.ndarray
        每个节点的社区 id（任意整数标签）。
    gamma : float
        分辨率参数。
    """
    A = np.asarray(A, dtype=np.float64)
    comm = np.asarray(comm)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise E100_INVALID_GRAPH("邻接矩阵必须为方阵")
    if np.any(A < -1e-15):
        raise E102_NEGATIVE_WEIGHT("邻接矩阵不可含负权")
    m = A.sum() / 2.0
    if m <= 0:
        return 0.0
    k = A.sum(axis=1)
    total = 0.0
    unique = np.unique(comm)
    for c in unique:
        mask = comm == c
        inc_c = A[np.ix_(mask, mask)].sum()  # 有序块内边权和
        tot_c = k[mask].sum()
        total += inc_c - gamma * (tot_c * tot_c) / (2.0 * m)
    return total / (2.0 * m)


def _louvain_one_level(A: np.ndarray, gamma: float, seed: int, max_passes: int, tol: float):
    """单层 Louvain 局部移动。

    Returns
    -------
    comm : np.ndarray
        当前层每个节点（聚合前）的最终社区标签（0..K-1 连续）。
    moved : bool
        本层是否有节点移动。
    """
    n = A.shape[0]
    k = A.sum(axis=1)
    m = A.sum() / 2.0
    if m <= 0:
        return np.arange(n), False
    comm = np.arange(n)
    tot = k.copy()  # 每个社区总度
    # inc_c: 块内有序边权和（初始为 0，单点社区无内部边）
    inc = np.zeros(n)

    rng = new_rng(seed)
    moved_any = False
    for _pass in range(max_passes):
        order = np.arange(n)
        rng.shuffle(order)
        improved = False
        for i in order:
            cur = comm[i]
            ki = k[i]
            if ki <= 0:
                continue
            # 候选社区 = 邻居社区 ∪ 当前社区
            row = A[i]
            nbr = row > 0
            cand_comm = np.unique(comm[nbr])
            if cur not in cand_comm:
                cand_comm = np.concatenate([cand_comm, [cur]])

            best_dq = 0.0  # 留在当前社区 dQ = 0
            best_c = cur

            # 当前社区内部边权（去掉 i 自身，对角已为 0）
            kin_cur = float(row[comm == cur].sum())  # Σ_{j in cur} A[i,j]
            # 移除 i 对当前社区的增量（常数，对所有候选相同）
            dq_removal = (-2.0 * kin_cur - gamma * ((tot[cur] - ki) ** 2 - tot[cur] ** 2) / (2.0 * m)) / (2.0 * m)

            for C in cand_comm:
                if C == cur:
                    dq = 0.0
                else:
                    kin_c = float(row[comm == C].sum())  # Σ_{j in C} A[i,j]
                    dq_add = (2.0 * kin_c - gamma * ((tot[C] + ki) ** 2 - tot[C] ** 2) / (2.0 * m)) / (2.0 * m)
                    dq = dq_add + dq_removal
                if dq > best_dq + tol:
                    best_dq = dq
                    best_c = int(C)

            if best_c != cur:
                # 应用移动：更新 inc / tot / comm
                inc[cur] -= 2.0 * kin_cur
                inc[best_c] += 2.0 * float(row[comm == best_c].sum())
                tot[cur] -= ki
                tot[best_c] += ki
                comm[i] = best_c
                improved = True
                moved_any = True
        if not improved:
            break
    # 重标号到连续 0..K-1
    _, comm = np.unique(comm, return_inverse=True)
    return comm, moved_any


def louvain(
    graph: GraphContainer,
    gamma: float = 1.0,
    seed: int = 0,
    max_levels: int | None = None,
    max_passes: int | None = None,
    move_tol: float | None = None,
) -> PartitionResult:
    """Louvain 社区发现（纯 NumPy）。

    Parameters
    ----------
    graph : GraphContainer
    gamma : float
        分辨率参数。
    seed : int
        随机种子（用于节点遍历顺序的确定性打乱）。

    Returns
    -------
    PartitionResult
        labels 为原始节点 → 最终社区 id；modularity 为最终 Q；
        meta["level_modularity"] 为每层模块度序列（用于单调不减门禁）。
    """
    from ..core.config import get_config

    cfg = get_config()
    if max_levels is None:
        max_levels = cfg.louvain_max_levels
    if max_passes is None:
        max_passes = cfg.louvain_max_passes
    if move_tol is None:
        move_tol = cfg.louvain_move_tol

    A = graph.adjacency.copy()
    n = A.shape[0]
    # mapping: 原始节点 -> 当前聚合节点 id
    mapping = np.arange(n)

    level_q: list[float] = []
    cur_A = A
    for level in range(max_levels):
        comm, moved = _louvain_one_level(cur_A, gamma, seed + level * 101, max_passes, move_tol)
        # 重映射原始节点
        mapping = comm[mapping]
        q = modularity(cur_A, comm, gamma)
        level_q.append(q)
        n_comm = int(comm.max()) + 1
        if not moved or n_comm == cur_A.shape[0]:
            break
        # 聚合
        K = n_comm
        new_A = np.zeros((K, K), dtype=np.float64)
        for i in range(cur_A.shape[0]):
            ci = comm[i]
            row = cur_A[i]
            nz = np.where(row > 0)[0]
            for j in nz:
                cj = comm[j]
                new_A[ci, cj] += row[j]
        cur_A = new_A
        if cur_A.shape[0] <= 1:
            break

    labels = mapping.copy()
    n_comm = int(labels.max()) + 1
    final_q = modularity(A, labels, gamma)
    return PartitionResult(
        labels=labels,
        n_communities=n_comm,
        modularity=final_q,
        score=final_q,
        meta={"level_modularity": level_q, "gamma": gamma, "seed": seed},
    )


def consolidate_to_k(A: np.ndarray, comm: np.ndarray, k: int, gamma: float = 1.0) -> np.ndarray:
    """凝聚式 k 归并：反复合并模块度增益最大的社区对，直至恰好剩 k 个社区。

    动机（分辨率极限修正）：标准模块度最大化会把大社区过度切分（Fortunato &
    Barthélemy 2007 的 resolution limit）。当目标社区数 k 已知时，通过按模块度增益
    贪心合并，可在保留 Louvain 多尺度结构的同时恢复正确粒度。

    合并增益（可证）：ΔQ(a,b) = W_ab/m − γ·tot_a·tot_b/(2m²)
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    _, comm = np.unique(comm, return_inverse=True)
    K = int(comm.max()) + 1
    if k >= K:
        return comm
    m = A.sum() / 2.0
    deg = A.sum(axis=1)

    # 社区级聚合矩阵：Agg[a,b] = 两社区间边权和（各计一次）；Agg[a,a] = 块内有序边权和
    Agg = np.zeros((K, K))
    for i in range(n):
        Agg[comm[i]] += np.bincount(comm, weights=A[i], minlength=K)
    tot = np.bincount(comm, weights=deg, minlength=K)

    active = np.ones(K, dtype=bool)
    parent = np.arange(K)  # 并查集：被合并的社区指向存活社区

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    while int(active.sum()) > k:
        idx = np.where(active)[0]
        best_gain = -np.inf
        best_pair = (idx[0], idx[1])
        for p in range(len(idx)):
            for q in range(p + 1, len(idx)):
                a, b = idx[p], idx[q]
                gain = Agg[a, b] / m - gamma * tot[a] * tot[b] / (2.0 * m * m)
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (a, b)
        a, b = best_pair
        # 合并 b -> a（并查集记录，保证节点映射不丢失）
        parent[b] = a
        Agg[a, a] = Agg[a, a] + Agg[b, b] + 2.0 * Agg[a, b]
        for c in idx:
            if c == a or c == b:
                continue
            Agg[a, c] += Agg[b, c]
            Agg[c, a] = Agg[a, c]
        tot[a] += tot[b]
        active[b] = False
        Agg[b, :] = 0.0
        Agg[:, b] = 0.0

    # 映射回节点：每个原始社区先解析到最终存活根，再重标号为 0..k-1
    roots = np.array([_find(c) for c in range(K)])
    new_id = np.full(K, -1, dtype=int)
    for j, c in enumerate(np.where(active)[0]):
        new_id[c] = j
    return new_id[roots[comm]]


def _aggregated_graph(A: np.ndarray, comm: np.ndarray):
    """按社区粗化原图，返回 (Agg, comm连续化, K)。Agg 对角置 0。"""
    _, comm = np.unique(comm, return_inverse=True)
    K = int(comm.max()) + 1
    Agg = np.zeros((K, K))
    for i in range(A.shape[0]):
        Agg[comm[i]] += np.bincount(comm, weights=A[i], minlength=K)
    np.fill_diagonal(Agg, 0.0)
    return Agg, comm, K


def consolidate_spectral(A: np.ndarray, comm: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """粗化图上的谱归并：在 Louvain 粗化后的社区图（K ≤ 数十）上做谱聚类。

    粗化起到去噪作用：社区图比原图干净得多，小图上的谱聚类既快又准。
    """
    from ..core.interfaces import GraphContainer
    from .spectral import spectral_clustering

    Agg, comm, K = _aggregated_graph(A, comm)
    if k >= K:
        return comm
    g = GraphContainer(n_nodes=K, adjacency=Agg, name="aggregated")
    res = spectral_clustering(g, k=k, seed=seed)
    return res.labels[comm]


def community_fuse(
    graph: GraphContainer,
    k: int | None = None,
    gamma: float = 1.0,
    seed: int = 0,
    n_restarts: int = 4,
) -> PartitionResult:
    """旗舰融合器 CommunityFuse —— 修复模块度分辨率极限的社区发现器。

    流程：
    1. 多起点 Louvain（不同遍历种子），取模块度最高者；
    2. 若给定目标社区数 k 且当前社区数 > k，生成两个归并候选：
       (a) 模块度增益贪心归并 consolidate_to_k
       (b) 粗化图上的谱归并 consolidate_spectral
       在**固定 k** 下按模块度选优（此时模块度可比，分辨率极限偏差消失）；
    3. 在原始图上做节点级局部精炼。

    注意：第 2 步的选优判据是模块度，但**仅在候选社区数相同（=k）时比较**；
    直接用模块度在「不同社区数」的划分间比较会偏向过度切分——那正是分辨率极限本身。
    """
    best: PartitionResult | None = None
    for r in range(n_restarts):
        res = louvain(graph, gamma=gamma, seed=seed + r * 7)
        if best is None or res.modularity > best.modularity:
            best = res

    A = graph.adjacency
    labels = best.labels
    chosen_strategy = "louvain_direct"
    if k is not None and best.n_communities > k:
        cand_q = consolidate_to_k(A, labels, k, gamma)
        cand_s = consolidate_spectral(A, labels, k, seed)
        mq = modularity(A, cand_q, gamma)
        ms = modularity(A, cand_s, gamma)
        if mq >= ms:
            labels, chosen_strategy = cand_q, "consolidate_modularity"
        else:
            labels, chosen_strategy = cand_s, "consolidate_spectral"

    refined = _refine_on_original(A, labels, gamma, seed, best.modularity)
    refined.meta["method"] = "CommunityFuse"
    refined.meta["strategy"] = chosen_strategy
    refined.meta["k_requested"] = k
    return refined


def _refine_on_original(
    A: np.ndarray, init_comm: np.ndarray, gamma: float, seed: int, base_q: float
) -> PartitionResult:
    """在原始图上以给定划分为初值局部移动精炼。"""
    n = A.shape[0]
    comm = init_comm.copy()
    # 连续化
    _, comm = np.unique(comm, return_inverse=True)
    k = A.sum(axis=1)
    m = A.sum() / 2.0
    # 计算每社区 tot / inc
    unique = np.unique(comm)
    tot = np.zeros(unique.max() + 1)
    inc = np.zeros(unique.max() + 1)
    for c in unique:
        mask = comm == c
        tot[c] = k[mask].sum()
        inc[c] = A[np.ix_(mask, mask)].sum()
    rng = new_rng(seed + 999)
    for _pass in range(50):
        order = np.arange(n)
        rng.shuffle(order)
        improved = False
        for i in order:
            cur = comm[i]
            ki = k[i]
            if ki <= 0:
                continue
            row = A[i]
            nbr = row > 0
            cand_comm = np.unique(comm[nbr])
            if cur not in cand_comm:
                cand_comm = np.concatenate([cand_comm, [cur]])
            best_dq = 0.0
            best_c = cur
            kin_cur = float(row[comm == cur].sum())
            dq_removal = (-2.0 * kin_cur - gamma * ((tot[cur] - ki) ** 2 - tot[cur] ** 2) / (2.0 * m)) / (2.0 * m)
            for C in cand_comm:
                if C == cur:
                    dq = 0.0
                else:
                    kin_c = float(row[comm == C].sum())
                    dq_add = (2.0 * kin_c - gamma * ((tot[C] + ki) ** 2 - tot[C] ** 2) / (2.0 * m)) / (2.0 * m)
                    dq = dq_add + dq_removal
                if dq > best_dq + 1e-12:
                    best_dq = dq
                    best_c = int(C)
            if best_c != cur:
                inc[cur] -= 2.0 * kin_cur
                inc[best_c] += 2.0 * float(row[comm == best_c].sum())
                tot[cur] -= ki
                tot[best_c] += ki
                comm[i] = best_c
                improved = True
        if not improved:
            break
    _, comm = np.unique(comm, return_inverse=True)
    q = modularity(A, comm, gamma)
    n_comm = int(comm.max()) + 1
    return PartitionResult(
        labels=comm,
        n_communities=n_comm,
        modularity=q,
        score=q,
        meta={"level_modularity": [base_q, q], "gamma": gamma, "seed": seed, "method": "CommunityFuse+refine"},
    )
