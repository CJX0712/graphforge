# graphForge 架构

> 作者：晨星

## 分层

```
cli.py / examples/run_demo.py        面向用户的入口与演示
        │
pipeline/pipeline.py                 评估链路：benchmark + 门禁
        │
graph/                               算法层
  build.py        图构造 / 拉普拉斯 / Fiedler / 连通分量
  community.py    Louvain / 模块度 / 归并 / CommunityFuse 旗舰
  spectral.py     谱聚类 / 对称归一化嵌入 / k-means
  kernels.py      Weisfeiler-Lehman 图核
  centrality.py   PageRank
  _eig.py         共享对称特征求解器（纯 NumPy，无 scipy）
        │
eval/metrics.py                     ARI / NMI / 变分信息
core/                               基础设施
  config.py       全局配置（环境变量覆盖 + schema 校验）
  errors.py       领域异常
  interfaces.py   GraphContainer / PartitionResult
  seed.py         确定性随机数
```

## 关键设计决策

### 1. 纯 NumPy 实现（零重依赖）
所有算法手写：谱分解用平移幂迭代 `(cI − M)`（c = λ_max + margin），其中 (cI−M) 半正定，其最大特征方向对应 M 的最小特征向量；配合收缩 + 修正 Gram-Schmidt，即使特征值聚集/退化也能得到正交特征向量。CI 在不装 scipy/networkx/sklearn 的隔离环境下仍跑通（`test_offline.py` 用 import 拦截器验证）。

### 2. CommunityFuse：修复模块度分辨率极限
Louvain 直接优化模块度 Q 会因分辨率极限（Fortunato & Barthélemy 2007）把大社区过度切分。旗舰器流程：
1. 多起点 Louvain（不同遍历种子），取模块度最高者；
2. 若给定目标社区数 k 且当前社区数 > k，生成两个归并候选：
   - **模块度增益贪心归并** `consolidate_to_k`（ΔQ(a,b) = W_ab/m − γ·tot_a·tot_b/(2m²)）
   - **粗化图上的谱归并** `consolidate_spectral`（社区图 K≤数十，谱聚类去噪）
3. 在**固定 k** 下按模块度无监督选优（此时模块度可比，分辨率偏差消失）；
4. 在原始图上做节点级局部精炼。

归并正确性用**并查集**记录合并父指针，保证节点映射不丢失（曾因未更新映射导致节点被打成 −1）。

### 3. 不变量交叉验证（信任可证量，不信任近似）
- 拉普拉斯对称半正定：最小特征值 ≥ −tol
- 谱嵌入正交：VᵀV = I ± 1e-16
- WL 同构：随机图与其置换副本的核值 == 自核
- 模块度单调：Louvain 跨层下降 ≤ 1e-9（实测 0.0）

### 4. 确定性
所有随机性经 `core/seed.py` 的 `seed_all` / `new_rng`，同 seed 两次 benchmark 核心指标逐位一致（CI 与 demo 双重验证）。

## 性能
- 谱嵌入：n≤300、k=4、2000 次迭代，单次约 0.5s（纯 NumPy）。
- 全 benchmark（15 runs）约 15–25s；pytest 全量 < 30s。
- 全系统仅依赖 NumPy，可在无网络的 CI 隔离环境运行。
