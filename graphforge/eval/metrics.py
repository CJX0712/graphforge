"""聚类/社区划分评测指标（纯 NumPy）。

作者：晨星

提供：ARI（调整兰德指数）、NMI（归一化互信息）、VI（信息变异）。
均支持任意整数标签（自动对齐类别集合）。
"""

from __future__ import annotations

import numpy as np


def _comb2(x):
    """组合数 C(x,2)，支持标量或数组（x<2 时为 0）。"""
    x = np.asarray(x, dtype=float)
    return np.where(x < 2.0, 0.0, x * (x - 1.0) / 2.0)


def contingency(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """列联表 C[true_label, pred_label]。"""
    true = np.asarray(true)
    pred = np.asarray(pred)
    tu = np.unique(true)
    tp = np.unique(pred)
    C = np.zeros((tu.size, tp.size), dtype=float)
    tmap = {v: i for i, v in enumerate(tu)}
    pmap = {v: j for j, v in enumerate(tp)}
    for t, p in zip(true, pred):
        C[tmap[t], pmap[p]] += 1.0
    return C


def ari(true: np.ndarray, pred: np.ndarray) -> float:
    """调整兰德指数（ARI），取值 [-1, 1]，随机划分≈0，完全一致=1。"""
    C = contingency(true, pred)
    n = C.sum()
    if n == 0:
        return 1.0
    a_i = C.sum(axis=1)
    b_j = C.sum(axis=0)
    index = float(np.sum(_comb2(C)))
    expected = float(np.sum(_comb2(a_i)) * np.sum(_comb2(b_j)) / _comb2(n))
    max_idx = 0.5 * (np.sum(_comb2(a_i)) + np.sum(_comb2(b_j)))
    if max_idx - expected < 1e-15:
        return 1.0 if np.isclose(index, expected, atol=1e-12) else 0.0
    return (index - expected) / (max_idx - expected)


def nmi(true: np.ndarray, pred: np.ndarray) -> float:
    """归一化互信息（几何平均归一化），取值 [0, 1]。"""
    C = contingency(true, pred)
    n = C.sum()
    if n == 0:
        return 1.0
    a_i = C.sum(axis=1)
    b_j = C.sum(axis=0)
    # MI
    mi = 0.0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            c = C[i, j]
            if c > 0:
                mi += c * np.log(n * c / (a_i[i] * b_j[j]))
    mi /= n
    h_true = -np.sum((a_i / n) * np.log(a_i / n + 1e-300))
    h_pred = -np.sum((b_j / n) * np.log(b_j / n + 1e-300))
    denom = np.sqrt(h_true * h_pred)
    if denom < 1e-15:
        return 1.0 if np.isclose(h_true, h_pred, atol=1e-12) else 0.0
    return mi / denom


def variation_of_information(true: np.ndarray, pred: np.ndarray) -> float:
    """信息变异 VI = H(true) + H(pred) - 2·MI，越小越一致。"""
    C = contingency(true, pred)
    n = C.sum()
    a_i = C.sum(axis=1)
    b_j = C.sum(axis=0)
    mi = 0.0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            c = C[i, j]
            if c > 0:
                mi += c * np.log(n * c / (a_i[i] * b_j[j]))
    mi /= n
    h_true = -np.sum((a_i / n) * np.log(a_i / n + 1e-300))
    h_pred = -np.sum((b_j / n) * np.log(b_j / n + 1e-300))
    return float(h_true + h_pred - 2.0 * mi)
