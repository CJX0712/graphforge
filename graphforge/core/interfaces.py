"""接口契约（Protocol）：保证模块可独立验证、可插拔。

作者：晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np


@dataclass
class GraphContainer:
    """不可变（构造后勿改）图容器。

    Attributes
    ----------
    n_nodes : int
        节点数。
    adjacency : np.ndarray
        对称、非负、对角为 0 的稠密邻接矩阵 (n_nodes, n_nodes)，float64。
    labels : np.ndarray | None
        可选节点属性标签（用于图核）。
    name : str
        可读名称。
    """

    n_nodes: int
    adjacency: np.ndarray
    labels: np.ndarray | None = None
    name: str = "graph"

    def __post_init__(self) -> None:
        a = self.adjacency
        if a.shape != (self.n_nodes, self.n_nodes):
            raise ValueError("adjacency 形状必须与 (n_nodes, n_nodes) 一致")
        if not np.allclose(a, a.T, atol=1e-12):
            raise ValueError("adjacency 必须对称")
        if np.any(a < -1e-15):
            raise ValueError("邻接矩阵不可含负权")
        # 强制对角为 0（简单图约定）
        np.fill_diagonal(a, 0.0)
        if self.labels is not None and self.labels.shape[0] != self.n_nodes:
            raise ValueError("labels 长度必须等于 n_nodes")


@dataclass
class PartitionResult:
    """社区划分 / 聚类结果。"""

    labels: np.ndarray  # 每个节点的最终社区 id (int)
    n_communities: int
    modularity: float = 0.0
    score: float = 0.0  # 方法特定主指标（如 ARI / NMI / kernel）
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRow:
    """单条 benchmark 记录。"""

    method: str
    metric: str
    value: float
    seed: int
    meta: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CommunityDetector(Protocol):
    """社区发现接口。"""

    def detect(self, graph: GraphContainer, seed: int) -> PartitionResult: ...


@runtime_checkable
class GraphKernel(Protocol):
    """图核接口：输入两图，输出核值（越大越相似）。"""

    def __call__(self, g1: GraphContainer, g2: GraphContainer) -> float: ...


@runtime_checkable
class Embedder(Protocol):
    """图嵌入接口：返回 (n_nodes, dim) 嵌入矩阵。"""

    def embed(self, graph: GraphContainer, seed: int) -> np.ndarray: ...
