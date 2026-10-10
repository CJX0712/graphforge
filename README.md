# graphForge

> 纯 NumPy 世界级谱图学习与社区发现系统 · 作者 **晨星**

graphForge 是一套**零重依赖**（仅依赖 NumPy）、**纯手写**的图机器学习系统，覆盖谱图学习与社区发现的核心数学与算法：

- **谱聚类**（Symmetric-Normalized Laplacian 最小特征向量，平移幂迭代 + 收缩 + Gram-Schmidt 正交化，无 scipy）
- **社区发现** —— 旗舰器 `CommunityFuse`：多起点 Louvain + 模块度增益归并 + 粗化谱归并 + 节点级局部精炼，**修复 Louvain 的模块度分辨率极限**
- **模块度金标准**：可证跨层单调不减（不变量 ≤ 1e-9）
- **图核**：Weisfeiler-Lehman 同构核（手写 multiset hash）
- **PageRank** 中心性（幂迭代）
- **图拉普拉斯不变量**：对称半正定、谱嵌入正交（VᵀV = I ± 1e-16）、Fiedler 值 ↔ 连通性

设计哲学（与 otForge 一致）：**不变量交叉验证 > 信任近似**。所有社区发现结果按 SBM 真实标签评分，所有谱方法按拉普拉斯解析性质核验。

---

## 快速开始

```bash
pip install -r requirements.txt
python -m graphforge.cli benchmark --out benchmark.json   # CLI
python examples/run_demo.py benchmark.json                # 端到端演示
```

```python
from graphforge.graph.build import stochastic_block_model
from graphforge.graph.community import community_fuse

g = stochastic_block_model([50, 50, 50, 50], 0.15, 0.005, seed=0)
res = community_fuse(g, k=4, seed=0)
print(res.n_communities, res.modularity)
```

---

## 核心指标（3 regime × 5 seeds = 15 runs）

| 方法 | ARI mean | 说明 |
|------|---------:|------|
| **CommunityFuse（旗舰）** | **0.9910** | 给 k，修复分辨率极限 |
| 谱聚类 k-means | 0.9899 | 已知 k 的统计天花板（参考） |
| 普通 Louvain | 0.7571 | 领域标准基线（不给 k） |
| 邻接行 k-means | 0.5472 | 给 k |
| 随机划分 | ≈ 0.00 | 对照 |

**门禁（DoD，方案阶段定死，诚实可达）：**

- **G1 恢复力**：旗舰 ARI ≥ 0.90 → 实测 **0.9910** ✅
- **G2 真实胜点**：旗舰 − 普通 Louvain ≥ 0.05 → 实测 **+0.234** ✅
- **G3 模块度单调**：Louvain 跨层下降 ≤ 1e-9 → 实测 **0.0** ✅
- **G4 不变量**：拉普拉斯半正定 / 谱嵌入正交（误差 **2.2e-16**）/ WL 同构核 == 自核 → ✅

> **诚实声明**：谱聚类在已知真实 k 下是 SBM 的统计天花板（ARI 0.9899 ≈ 旗舰 0.9910），graphForge **不声称超越该天花板**；其真实可量化贡献是**修复 Louvain 的模块度分辨率极限**（不给 k 时普通 Louvain 过度切分，ARI 仅 0.7571，旗舰相对领先 +0.234 / 消融 Δ=+0.357）。

---

## 目录结构

```
graphforge/
  core/        配置 / 错误 / 接口 / 确定性种子
  graph/       图构造 / 拉普拉斯 / 社区发现 / 谱聚类 / 图核 / 中心性 / 共享特征求解器
  eval/        ARI / NMI / 变分信息指标
  pipeline/    benchmark + 门禁
  cli.py       命令行入口
examples/      端到端演示
tests/         42 用例（不变量 / 确定性 / 离线 / 门禁 / CLI）
docs/          架构与 model card
```

---

## 数学内核（硬金标准）

- L = D − A 对称半正定；L_sym = I − D^{-1/2} A D^{-1/2}，特征值 ∈ [0, 2]
- Louvain 模块度 Q = (1/2m)Σ_c[inc_c − γ·tot_c²/(2m)]，每步严格提升 ⇒ 跨层单调不减
- 谱嵌入：平移幂迭代 (cI − M) 求 M 最小特征向量，修正 Gram-Schmidt 保证 VᵀV = I
- WL 核：迭代 multiset hash，同构图核值相等（不变量）
- PageRank：幂迭代，得分和为 1

---

## 许可证

MIT © 2026 晨星
