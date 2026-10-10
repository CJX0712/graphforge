"""共享对称特征值求解器（纯 NumPy，无 scipy）。

作者：晨星

方法：平移幂迭代 (cI − M) + 收缩 + 修正 Gram-Schmidt 正交化。
- c = λ_max(M) + margin ⇒ (cI − M) 半正定，其最大特征值对应 M 的最小特征值。
- 每次迭代对已求得方向做两轮 Gram-Schmidt，保证 V^T V = I（即使特征值聚集/退化）。
"""

from __future__ import annotations

import numpy as np

from ..core.seed import new_rng


def power_iter_largest(M: np.ndarray, iters: int, seed: int) -> float:
    """对称矩阵最大特征值（幂迭代，Rayleigh 商）。"""
    n = M.shape[0]
    if n == 0:
        return 0.0
    rng = new_rng(seed)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v) + 1e-300
    for _ in range(iters):
        w = M @ v
        nm = np.linalg.norm(w)
        if nm < 1e-300:
            break
        v = w / nm
    return float(v @ (M @ v))


def symeig_smallest(
    M: np.ndarray, k: int, iters: int = 500, seed: int = 0, margin: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """求对称矩阵 M 最小的 k 个特征值及其特征向量。

    Returns
    -------
    V : (n, k)  列为特征向量（正交归一）
    lam : (k,)  特征值（升序）
    """
    n = M.shape[0]
    k = max(0, min(k, n))
    if k == 0:
        return np.zeros((n, 0)), np.zeros(0)

    lam_max = power_iter_largest(M, iters, seed)
    c = lam_max + margin
    Mcur = M.copy()
    rng = new_rng(seed + 1)
    vecs: list[np.ndarray] = []
    lambs: list[float] = []

    for _ in range(k):
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v) + 1e-300
        for _ in range(iters):
            w = c * v - (Mcur @ v)
            # 修正 Gram-Schmidt：投影掉已求得的方向（两轮，数值稳定）
            for _gs in range(2):
                for u in vecs:
                    w -= (u @ w) * u
            nm = np.linalg.norm(w)
            if nm < 1e-300:
                break
            w = w / nm
            if np.linalg.norm(w - v) < 1e-14:
                v = w
                break
            v = w
        lam = float(v @ (M @ v))  # 对原矩阵的 Rayleigh 商
        vecs.append(v)
        lambs.append(lam)
        Mcur = Mcur - lam * np.outer(v, v)

    order = np.argsort(lambs)
    V = np.column_stack([vecs[i] for i in order])
    lam = np.array([lambs[i] for i in order])
    return V, lam
