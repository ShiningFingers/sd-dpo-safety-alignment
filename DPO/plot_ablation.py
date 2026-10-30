r"""plot_ablation.py —— 消融实验：精确 p 值 + fig5 曲线图"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs("fig", exist_ok=True)

b   = np.load("s_base_out.npy")
eps = [10, 30, 50]
d   = {e: np.load(f"s_dpo_out_epoch_{e}.npy") for e in eps}

print("=== A. 各 checkpoint vs baseline（配对，n=150，可直接抄进报告）===")
for e in eps:
    diff = d[e] - b
    t, p = stats.ttest_rel(d[e], b)
    coh  = diff.mean() / diff.std(ddof=1)
    print(f"epoch_{e:<3} mean={d[e].mean():+.4f}  Δ={diff.mean():+.4f}  "
          f"t={t:+.4f}  df=149  p={p:.3e}  d={coh:+.4f}")

print("\n=== B. checkpoint 两两比较（Bonferroni α'=0.0167）===")
for a, c in [(30, 10), (50, 10), (30, 50)]:
    diff = d[a] - d[c]
    t, p = stats.ttest_rel(d[a], d[c])
    r1 = "显著" if p < 0.05 else "不显著"
    r2 = "通过" if p < 0.05/3 else "未通过"
    print(f"epoch_{a} vs epoch_{c}: Δ={diff.mean():+.4f}  t={t:+.4f}  "
          f"df=149  p={p:.4e}  [{r1}]  Bonferroni:{r2}")

# ---------- 画图 ----------
x     = [0] + eps
means = [b.mean()] + [d[e].mean() for e in eps]
ci    = [1.96*s.std(ddof=1)/np.sqrt(len(s)) for s in [b] + [d[e] for e in eps]]
lbl   = ['Baseline', 'epoch_10', 'epoch_30', 'epoch_50']

fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
ax.errorbar(x, means, yerr=ci, fmt='-o', capsize=6, color='#C00000',
            ecolor='#C00000', markersize=9, linewidth=2.2)
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xticks(x); ax.set_xticklabels(lbl)
ax.set_xlabel('训练轮数 (epoch)')
ax.set_ylabel('Safety Gain 均值（集外 n=150）')
ax.set_title('训练轮数消融：DPO 安全性增益随 epoch 的变化（误差棒=95% CI）')

for xi, mi in zip(x, means):
    ax.text(xi, mi - 1.0, f'{mi:+.4f}', ha='center', fontsize=9)
i = int(np.argmax(means))
ax.annotate(f'峰值 {means[i]:+.4f}', xy=(x[i], means[i]),
            xytext=(x[i], means[i] + 1.3), ha='center',
            fontsize=11, color='#C00000',
            arrowprops=dict(arrowstyle='->', color='#C00000'))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig/fig5_ablation.png", dpi=150, bbox_inches='tight')
print("\n已保存: fig/fig5_ablation.png")