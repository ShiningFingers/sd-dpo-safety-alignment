# SD-DPO Safety Alignment

基于 Stable Diffusion v1.5 的文生图安全对齐实验：使用 **DPO（直接偏好优化）+ LoRA** 对模型进行安全偏好对齐，并以 **CLIP Score** 为指标、**配对 t 检验**验证效果。在仅训练约 0.1% 参数的条件下，模型在暴力类提示词下的安全性倾向由负转正，且在未见概念上显著泛化。

## 核心结果

| 分组 | n | Baseline Gain | DPO Gain | Δ | t | p | Cohen's d |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 训练集内 | 50 | −9.65 | +0.88 | +10.53 | 14.22 | 5.1×10⁻¹⁹ | 2.01 |
| 训练集外（主结论） | 150 | −5.24 | +1.06 | +6.31 | 14.33 | 8.1×10⁻³⁰ | 1.17 |

Gain = CLIP safe score − unsafe score，正值表示图像偏向安全。两组 p ≪ 0.05 拒绝原假设，Cohen's d 均大于 0.8（大效应）。训练集外同样显著，说明对齐效果可泛化到未见过的暴力概念。

## 方法概览

- **基座模型**：Stable Diffusion v1.5（约 8 亿参数）
- **微调方式**：LoRA，冻结 UNet 主体，仅在注意力层注入低秩旁路，可训练参数 **797,184 个（≈ 总参数 0.1%）**，LoRA 权重文件 3.2 MB
- **优化目标**：DPO 损失，β = 1000.0（强偏好约束），同时拉高安全图生成概率、压低不安全图生成概率
- **训练数据**：50 对"暴力/安全"图文偏好对（由原始 SD 生成）
- **超参**：50 epochs，batch size = 1，AdamW lr = 5e-5，单卡（RTX 4060 8GB，float16）约 69 分钟
- **训练曲线**：DPO 损失 0.69 → 0.078，隐式准确率 0.20 → 0.98
- **评估**：40 条提示词（10 集内 + 30 集外）× 5 随机种子，同提示词同种子下 Baseline 与 DPO 各生成一图，构成配对样本

## 目录结构

| 路径 | 说明 |
|:--|:--|
| `DPO/` | 主实验：代码、数据、权重、图表 |
| `DPO/data/generated/` | 100 张训练偏好图（50 安全 + 50 暴力） |
| `DPO/fig/` | 报告统计图（fig1–fig5） |
| `DPO/lora_safety_aligned/` | 最终 LoRA 权重 |
| `DPO/lora_safety_aligned_epoch_{10,30,50}/` | 消融实验用 checkpoint |
| `DPO/*.py` | 训练、评估、画图、消融脚本 |
| `DPO/*.npy` | CLIP 打分原始数组（画图数据源） |
| `DPO/data.csv` | 50 对偏好数据清单 |
| `DPO/requirements.txt` | Python 依赖 |
| `SD/` | 独立的 SD LoRA 微调练习（狐狸图像，非本实验） |
| `教程/` | 五份面向零基础的实验拆解教程 |
| `DPO安全对齐实验报告.md` | 正式实验报告 |
| `DPO安全对齐实验完整记录.md` | 技术学习记录（原理、代码、踩坑） |
| `DPO实验说明书.docx` | 课程实验手册 |
| `GitHub上传踩坑记录.md` | 上传 GitHub 的经验笔记 |

## 复现步骤

```bash
# 1. 环境
conda create -n dpo python=3.10
conda activate dpo
pip install -r DPO/requirements.txt

# 2. 训练（基座模型 SD v1.5 需自行下载到本地并修改 DPO/eval_dpo.py 中的 BASE_MODEL 路径）
cd DPO
python mini_train.py        # 训练 50 轮，产出 lora_safety_aligned 及中间 checkpoint

# 3. 对比评估
python eval_dpo.py          # 生成 Baseline 与 DPO 两组图，CLIP 打分，配对 t 检验

# 4. 画图
python plot_results.py      # fig1–fig3
python plot_triple_split.py # fig4
python ablation.py && python stat_ablation.py && python plot_ablation.py  # fig5 消融
```

> 仓库已包含 `data/generated/`（训练偏好图）、`fig/`（报告图）、4 个 LoRA 权重目录及 `.npy` 分数文件，可直接用于核对报告结果，无需重跑。

## 文档导航

- [实验报告](DPO安全对齐实验报告.md) — 正式报告，含统计分析与结论
- [完整记录](DPO安全对齐实验完整记录.md) — 原理、代码逐条讲解、踩坑记录
- [实验说明书](DPO实验说明书.docx) — 课程下发的实验手册
- [教程目录](教程/) — 零基础拆解：全景、四大概念、代码逐行、统计检验、实验精髓
- [上传踩坑记录](GitHub上传踩坑记录.md) — GitHub 推送经验（代理、HTTP/1.1、PAT）

## 局限与后续

- **概念覆盖窄**：仅对齐"暴力"单一概念，原版 AlignGuard 覆盖 723 概念 / 7 类别
- **长尾失效**：训练集外约 2% 样本（3/150）对齐未生效，unsafe 标准差反而增大
- **副作用**：safe 分数同步下降（集外 −5.39），语义相关性降低，泛化场景代价更明显
- **评价指标单一**：仅 CLIP Score 代理指标，无人工评测
- **过拟合迹象**：消融显示 epoch_30（Gain +1.54）优于 epoch_50（+1.06），训练轮数过多存在增益回落

## 关于仓库体积

本仓库未包含 `DPO/eval_baseline/`、`DPO/eval_dpo/`、`DPO/test_40_concepts/` 等中间生成图（约 360 MB）。这些图可由 `eval_dpo.py` / `test.py` 重新生成，不影响实验复现。
