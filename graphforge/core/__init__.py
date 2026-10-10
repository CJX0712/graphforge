"""core 子包：类型、错误、配置、接口、确定性种子。"""

from .config import Config
from .errors import (
    E100_INVALID_GRAPH,
    E101_NOT_SYMMETRIC,
    E102_NEGATIVE_WEIGHT,
    E200_EMPTY_GRAPH,
    E300_NOT_CONNECTED,
    E400_CONVERGENCE,
    GraphForgeError,
)
from .seed import set_all

__all__ = [
    "GraphForgeError",
    "E100_INVALID_GRAPH",
    "E101_NOT_SYMMETRIC",
    "E102_NEGATIVE_WEIGHT",
    "E200_EMPTY_GRAPH",
    "E300_NOT_CONNECTED",
    "E400_CONVERGENCE",
    "Config",
    "set_all",
]
