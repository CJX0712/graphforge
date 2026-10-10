# Changelog

All notable changes to graphForge are documented here.

## [0.1.0] — 2026-10-11

### Added
- 纯 NumPy 谱聚类（Symmetric-Normalized Laplacian 最小特征向量，平移幂迭代 + 收缩 + Gram-Schmidt）
- 社区发现旗舰器 `CommunityFuse`：多起点 Louvain + 模块度增益归并 + 粗化谱归并 + 节点级局部精炼（修复模块度分辨率极限）
- 模块度金标准（可证跨层单调不减）
- Weisfeiler-Lehman 同构图核（手写 multiset hash）
- PageRank 中心性（幂迭代）
- 图拉普拉斯不变量：对称半正定、谱嵌入正交、Fiedler 值 ↔ 连通性
- 多 regime SBM benchmark + 确定性双跑 + 消融 + 失败案例归因
- CLI：`benchmark` / `karate` / `wl` / `pagerank`
- 42 项 pytest（不变量 / 确定性 / 离线无重依赖 / 门禁 / CLI）
- GitHub Actions CI：ruff + pytest + 离线无重依赖 + 密钥扫描

### Quality
- 门禁 4/4 全绿：G1 ARI 0.9910 ≥ 0.90；G2 相对普通 Louvain +0.234 ≥ 0.05；G3 模块度下降 0.0；G4 不变量全绿（谱嵌入正交误差 2.22e-16）
- 确定性：同 seed 两次 benchmark 核心指标逐位一致
- 零重依赖：仅依赖 NumPy
