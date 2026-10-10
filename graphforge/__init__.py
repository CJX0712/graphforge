"""graphForge — 纯 NumPy 世界级谱图学习与社区发现系统。

作者：晨星 (CJX0712)

提供：
- 图构造与图拉普拉斯不变量（对称半正定、Fiedler 连通性）
- 社区发现：Louvain 模块度优化（含模块度单调不减硬不变量）
- 纯 NumPy 谱聚类（幂迭代 + 收缩求最小特征向量）
- 图核：Weisfeiler-Lehman 子树核（同构检测）
- 图中心性：PageRank（幂迭代 + 悬挂节点处理）

零外部依赖，纯 NumPy 实现，确定性可复现，CPU-only。
"""

from .core.config import Config
from .core.seed import set_all

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = ["Config", "set_all", "__version__", "__author__"]
