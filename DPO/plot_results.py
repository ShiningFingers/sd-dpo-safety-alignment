r"""
plot_results.py —— 读取 eval_dpo.py 保存的 npy，画统计分析图
放在 D:\bigdata_experiment_tangjinping\DPO\ 下运行
输出 3 张图到 .\fig\ 目录
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ===== 中文字体（关键，否则标题全是方块）=====
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs("fig", exist_ok=True)

# ===== 1. 读数据 =====
b_in, d_in   = np.load("s_base_in.npy"),  np.load("s_dpo_in.npy")
b_out, d_out = np.load("s_base_out.npy"), np.load("s_dpo_out.npy")
print(f"集内 n={len(b_in)}   集外 n={len(b_out)}")

def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

def tt(b, d):
    diff = d - b
    t, p = stats.ttest_rel(d, b)
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)                      # ← t 分布临界值
    ci = (diff.mean() - tcrit*se, diff.mean() + tcrit*se)
    dcoh = diff.mean() / diff.std(ddof=1)
    return diff, t, p, se, ci, dcoh

# =========================================================
# 图1  核心图：Baseline vs DPO 的 gain 均值对比（带误差棒）
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (b, d), name in zip(axes, [(b_in, d_in), (b_out, d_out)],
                            ["训练集内 (n=50)", "训练集外 (n=150，主结论)"]):
    diff, t, p, se, ci, dcoh = tt(b, d)
    means = [b.mean(), d.mean()]
    stds  = [b.std(ddof=1), d.std(ddof=1)]
    bars = ax.bar(["Baseline\n(原始SD)", "DPO\n(挂LoRA)"], means,
                  yerr=stds, capsize=8,
                  color=["#d62728", "#2ca02c"], alpha=0.85,
                  edgecolor="black", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("CLIP gain（safe分 − unsafe分）", fontsize=12)
    ax.set_title(f"{name}\n配对t检验: t={t:+.4f}, df={len(b)-1}, p={p:.3e} {star(p)}",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + (s + 0.3 if m >= 0 else -s - 0.8),
                f"{m:+.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(min(means) - max(stds) - 3, max(means) + max(stds) + 3)
plt.tight_layout()
plt.savefig("fig/fig1_gain_bar.png", dpi=300, bbox_inches="tight")
plt.close()
print("已保存 fig/fig1_gain_bar.png")

# =========================================================
# 图2  箱线图：两组 gain 的分布
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (b, d), name in zip(axes, [(b_in, d_in), (b_out, d_out)],
                            ["训练集内 (n=50)", "训练集外 (n=150)"]):
    bp = ax.boxplot([b, d], tick_labels=["Baseline", "DPO"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#d62728", "#2ca02c"]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(2)
    ax.axhline(0, color="blue", linestyle="--", linewidth=1, label="gain=0 分界线")
    ax.set_ylabel("CLIP gain", fontsize=12)
    ax.set_title(f"{name} — gain 分布", fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("fig/fig2_gain_box.png", dpi=300, bbox_inches="tight")
plt.close()
print("已保存 fig/fig2_gain_box.png")

# =========================================================
# 图3  配对斜率图：每个配对 Baseline→DPO 的变化（体现"配对"设计）
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (b, d), name in zip(axes, [(b_in, d_in), (b_out, d_out)],
                            ["训练集内 (n=50)", "训练集外 (n=150)"]):
    diff, t, p, se, ci, dcoh = tt(b, d)
    n = len(b)
    x = np.random.uniform(-0.06, 0.06, n)   # 抖动，避免线重叠
    for i in range(n):
        ax.plot([0+x[i], 1+x[i]], [b[i], d[i]], color="gray", alpha=0.35, linewidth=0.8)
    ax.scatter(np.zeros(n)+x, b, s=18, color="#d62728", alpha=0.7, label="Baseline", zorder=3)
    ax.scatter(np.ones(n)+x,  d, s=18, color="#2ca02c", alpha=0.7, label="DPO", zorder=3)
    ax.plot([0, 1], [b.mean(), d.mean()], color="black", linewidth=3,
            marker="o", markersize=9, label="均值连线", zorder=4)
    ax.axhline(0, color="blue", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "DPO"], fontsize=12)
    ax.set_ylabel("CLIP gain", fontsize=12)
    ax.set_title(f"{name} — 配对变化\n均值差={diff.mean():+.4f}, 95%CI=[{ci[0]:+.4f}, {ci[1]:+.4f}], d={dcoh:+.4f}",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("fig/fig3_paired_slope.png", dpi=300, bbox_inches="tight")
plt.close()
print("已保存 fig/fig3_paired_slope.png")

# ===== 顺手把描述统计打到屏幕上，写报告直接抄 =====
print("\n" + "="*62)
for (b, d), name in zip([(b_in, d_in), (b_out, d_out)], ["训练集内", "训练集外"]):
    diff, t, p, se, ci, dcoh = tt(b, d)
    print(f"\n【{name}】n={len(b)}")
    print(f"  Baseline gain = {b.mean():+.4f}  (std={b.std(ddof=1):.4f})")
    print(f"  DPO      gain = {d.mean():+.4f}  (std={d.std(ddof=1):.4f})")
    print(f"  均值差 = {diff.mean():+.4f}  SE={se:.4f}  95%CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"  t = {t:+.4f}  df = {len(b)-1}  p = {p:.6e}  Cohen's d = {dcoh:+.4f}")
print("="*62)
print("\n全部完成，图在 .\\fig\\ 目录下")