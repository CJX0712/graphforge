"""全局配置：环境变量覆盖 + schema 校验。

作者：晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    """graphForge 全局配置。

    所有数值均带合理默认；支持通过环境变量 ``GF_*`` 覆盖，并在构造时做范围校验。
    """

    # ---- 确定性 ----
    default_seed: int = 20261010

    # ---- 社区发现 (Louvain) ----
    louvain_max_levels: int = 100
    louvain_max_passes: int = 100
    louvain_move_tol: float = 1e-12
    louvain_resolution: float = 1.0

    # ---- 谱聚类 ----
    spectral_power_iters: int = 2000
    spectral_deflation_iters: int = 50
    spectral_tol: float = 1e-10
    spectral_kmeans_iters: int = 300
    spectral_kmeans_tol: float = 1e-9

    # ---- 图核 (Weisfeiler-Lehman) ----
    wl_max_iter: int = 5

    # ---- PageRank ----
    pagerank_alpha: float = 0.85
    pagerank_iters: int = 500
    pagerank_tol: float = 1e-12

    # ---- 门禁 (DoD) ----
    gate_min_recovery_ari: float = 0.90  # SBM 强结构下旗舰 ARI 均值下限
    gate_min_margin_over_baseline: float = 0.05  # 旗舰 ARI 相对谱聚类基线的最小领先
    gate_max_modularity_decrease: float = 1e-9  # Louvain 跨层模块度下降上限（应为 0）
    gate_max_inv_violation: float = 1e-9  # 不变量违反上限

    def __post_init__(self) -> None:
        env_map: Dict[str, str] = {
            "GF_DEFAULT_SEED": "default_seed",
            "GF_LOUVAIN_MAX_LEVELS": "louvain_max_levels",
            "GF_LOUVAIN_RESOLUTION": "louvain_resolution",
            "GF_SPECTRAL_POWER_ITERS": "spectral_power_iters",
            "GF_PAGERANK_ALPHA": "pagerank_alpha",
            "GF_WL_MAX_ITER": "wl_max_iter",
        }
        for env_key, attr in env_map.items():
            if env_key in os.environ:
                try:
                    cur = getattr(self, attr)
                    if isinstance(cur, bool):
                        setattr(self, attr, os.environ[env_key].lower() in ("1", "true", "yes"))
                    elif isinstance(cur, int):
                        setattr(self, attr, int(os.environ[env_key]))
                    else:
                        setattr(self, attr, float(os.environ[env_key]))
                except ValueError:
                    raise ValueError(f"环境变量 {env_key} 无法解析为 {type(getattr(self, attr)).__name__}")

        # ---- schema 校验 ----
        if self.default_seed < 0:
            raise ValueError("default_seed 必须 >= 0")
        if not (0.0 < self.louvain_resolution <= 10.0):
            raise ValueError("louvain_resolution 应在 (0, 10]")
        if self.louvain_max_levels < 1:
            raise ValueError("louvain_max_levels 必须 >= 1")
        if not (0.0 < self.pagerank_alpha < 1.0):
            raise ValueError("pagerank_alpha 应在 (0, 1)")
        if not (0.0 <= self.gate_min_recovery_ari <= 1.0):
            raise ValueError("gate_min_recovery_ari 应在 [0, 1]")

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def load(cls, overrides: Dict[str, Any] | None = None) -> "Config":
        cfg = cls()
        if overrides:
            for k, v in overrides.items():
                if not hasattr(cfg, k):
                    raise AttributeError(f"未知配置项: {k}")
                setattr(cfg, k, v)
        return cfg


# 模块级单例（可被 set_all 重置默认值）
_DEFAULT = Config()


def get_config() -> Config:
    return _DEFAULT


def reset_config(cfg: Config | None = None) -> Config:
    global _DEFAULT
    _DEFAULT = cfg or Config()
    return _DEFAULT
