"""图核：Weisfeiler–Lehman 子树核（纯 NumPy）。

作者：晨星

数学内核（硬金标准）：
- WL 通过颜色细化（1-WL）将每个节点迭代标记为邻居标签的多重集哈希。
- 同构图 ⇒ 相同特征向量 ⇒ kernel(G, H) = kernel(G, G) = kernel(H, H)。
- 非同构（多数情形）⇒ kernel(G, H) < kernel(G, G)。
- 共享词表保证跨图一致编号：相同子树子结构映射到同一整数 id。
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..core.interfaces import GraphContainer


def _run_wl(graphs: List[GraphContainer], n_iter: int) -> List[np.ndarray]:
    """对一组图运行 WL，返回每图跨层拼接的计数特征向量（共享词表）。

    词表构建顺序确定（按图顺序、节点顺序、迭代顺序），保证同构映射一致。
    """
    vocab: dict[str, int] = {}
    all_level_ids: List[List[np.ndarray]] = []

    for g in graphs:
        A = g.adjacency
        n = A.shape[0]
        if g.labels is not None:
            init = np.asarray(g.labels, dtype=int).copy()
        else:
            init = A.sum(axis=1).astype(int)
        # 初始层：把初始标签纳入词表
        cur = init.copy()
        for i in range(n):
            ks = f"I|{int(cur[i])}"
            if ks not in vocab:
                vocab[ks] = len(vocab)
            cur[i] = vocab[ks]
        level_ids: List[np.ndarray] = [cur.copy()]

        for _ in range(n_iter):
            keys = []
            for i in range(n):
                nbrs = np.where(A[i] > 0)[0]
                nb = tuple(sorted(int(cur[j]) for j in nbrs))
                keys.append((int(cur[i]), nb))
            new = np.empty(n, dtype=int)
            for i, key in enumerate(keys):
                ks = f"{key[0]}|" + (",".join(str(x) for x in key[1]))
                if ks not in vocab:
                    vocab[ks] = len(vocab)
                new[i] = vocab[ks]
            cur = new
            level_ids.append(cur.copy())
        all_level_ids.append(level_ids)

    V = len(vocab)
    vecs: List[np.ndarray] = []
    for level_ids in all_level_ids:
        vec = np.zeros(V, dtype=float)
        for ids in level_ids:
            for lab in ids:
                vec[lab] += 1.0
        vecs.append(vec)
    return vecs


def wl_features(graph: GraphContainer, n_iter: int = 5) -> np.ndarray:
    """返回单图的 WL 跨层拼接计数特征向量。"""
    return _run_wl([graph], n_iter)[0]


def weisfeiler_lehman_kernel(g1: GraphContainer, g2: GraphContainer, n_iter: int = 5) -> float:
    """Weisfeiler–Lehman 子树核值（越大越相似）。

    同构图返回 kernel(G, G)；非同构通常严格小于自核。
    """
    vecs = _run_wl([g1, g2], n_iter)
    return float(np.dot(vecs[0], vecs[1]))
