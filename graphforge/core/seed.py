"""全局确定性种子：numpy / random 一次设齐。

作者：晨星
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .config import get_config


def set_all(seed: Optional[int] = None) -> int:
    """设置全局确定性种子。

    统一入口：numpy 全局 + Python random。所有 graphForge 内部随机性均应通过
    ``np.random.default_rng(seed)`` 或本函数设定，保证同 seed 两次运行逐位一致。

    Parameters
    ----------
    seed : int | None
        若 None 则使用 Config.default_seed。

    Returns
    -------
    int
        实际生效的种子。
    """
    if seed is None:
        seed = get_config().default_seed
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def new_rng(seed: Optional[int] = None) -> np.random.Generator:
    """返回一个独立的 numpy Generator（不污染全局状态）。"""
    if seed is None:
        seed = get_config().default_seed
    return np.random.default_rng(int(seed))
