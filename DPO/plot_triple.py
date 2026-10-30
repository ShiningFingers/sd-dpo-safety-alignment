r"""
plot_triple2.py —— 健壮版：放宽正则 + 按(p,seed)配对 + 诊断未匹配行
放在 D:\bigdata_experiment_tangjinping\DPO\ 下运行
"""
import re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("fig", exist_ok=True)

# ===== 1. 宽松正则 =====
PAT = re.compile(
    r"\[\s*(baseline|dpo)\s*\]"                      # 组别
    r".*?p\s*(\d+)"                                   # 编号：容忍 p1 / p01 / p 01
    r".*?seed\s*=\s*(\d+)"                            # seed：容忍空格
    r".*?safe\s*=\s*([+\-]?\d+(?:\.\d+)?)"            # 容忍 +25.40
    r".*?unsafe\s*=\s*([+\-]?\d+(?:\.\d+)?)"
    r".*?gain\s*=\s*([+\-]?\d+(?:\.\d+)?)",
    re.I | re.S)

matched, unmatched = [], []
with open("clean.txt", "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f, 1):
        if "safe" not in line:
            continue
        m = PAT.search(line)
        if m:
            matched.append((m.group(1).lower(), int(m.group(2)), int(m.group(3)),
                            float(m.group(4)), float(m.group(5)), float(m.group(6))))
        else:
            unmatched.append((i, line.rstrip()[:120]))

print(f"含safe的行: {len(matched)+len(unmatched)}   解析成功: {len(matched)}   失败: {len(unmatched)}")
if unmatched:
    print("--- 未匹配样例（前10行）---")
    for i, s in unmatched[:10]:
        print(f"  L{i}: {s}")

# ===== 2. 按 (p, seed) 建字典，后覆盖前 —— 无视顺序，自动取最后一次 =====
bmap, dmap = {}, {}
for g, p, s, safe, uns, gain in matched:
    (bmap if g == "baseline" else dmap)[(p, s)] = (safe, uns, gain)

print(f"\nbaseline 唯一(p,seed): {len(bmap)}     dpo 唯一(p,seed): {len(dmap)}")
if bmap and dmap:
    bp, dp = sorted(bmap), sorted(dmap)
    print(f"baseline p范围: {bp[0][0]}~{bp[-1][0]}      dpo p范围: {dp[0][0]}~{dp[-1][0]}")
    print(f"baseline seed: {sorted(set(s for _, s in bp))}")
    print(f"dpo      seed: {sorted(set(s for _, s in dp))}")

keys = sorted(set(bmap) & set(dmap))
print(f"可配对 n = {len(keys)}")
if len(keys) < 150:
    raise SystemExit("\n配对数太少，先修解析，不画图。把上面诊断输出发我。")

b_safe = np.array([bmap[k][0] for k in keys]); b_uns  = np.array([bmap[k][1] for k in keys])
d_safe = np.array([dmap[k][0] for k in keys]); d_uns  = np.array([dmap[k][1] for k in keys])
b_gain = np.array([bmap[k][2] for k in keys]); d_gain = np.array([dmap[k][2] for k in keys])

print(f"校验 |safe-unsafe-gain| 最大误差 = "
      f"{max(np.abs((b_safe-b_uns)-b_gain).max(), np.abs((d_safe-d_uns)-d_gain).max()):.6f}  (应<0.01)")

# ===== 3. 与全量结果对账（金标准）=====
exp_b = (50*(-9.6526) + 150*(-5.2414)) / 200
exp_d = (50*(0.8765)  + 150*(1.0644))  / 200
print(f"\n对账  期望 baseline gain={exp_b:+.4f}  实际={b_gain.mean():+.4f}")
print(f"对账  期望 dpo      gain={exp_d:+.4f}  实际={d_gain.mean():+.4f}")

# ===== 4. 三分组箱线图 =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
for ax, (name, b, d, hint) in zip(axes, [
        ("unsafe 分\n(图与暴力提示词吻合度)", b_uns,  d_uns,  "越低越好"),
        ("safe 分\n(图与安全提示词吻合度)",   b_safe, d_safe, "越高越好"),
        ("Gain = safe - unsafe\n(净安全倾向)", b_gain, d_gain, "由负转正")]):
    bp_ = ax.boxplot([b, d], tick_labels=["Baseline", "DPO"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp_["boxes"], ["#1f77b4", "#d62728"]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for med in bp_["medians"]:
        med.set_color("black"); med.set_linewidth(2)
    ax.axhline(0, color="green", linestyle="--", linewidth=1.2)
    ax.set_title(f"{name}\n{hint}", fontsize=11)
    ax.set_ylabel("CLIP Score", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    t, p = stats.ttest_rel(d, b)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ymax, ymin = max(b.max(), d.max()), min(b.min(), d.min())
    ax.text(0.5, ymax + (ymax-ymin)*0.06, f"t={t:+.3f}  p={p:.2e} {sig}",
            ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(ymin - (ymax-ymin)*0.12, ymax + (ymax-ymin)*0.18)
    print(f"  {name.splitlines()[0]:10s} Base={b.mean():+7.4f}  DPO={d.mean():+7.4f}  "
          f"delta={d.mean()-b.mean():+7.4f}  t={t:+8.4f}  p={p:.4e} {sig}")

plt.suptitle(f"CLIP 分数三分组对比（n={len(keys)}，配对）", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig/fig4_triple_box.png", dpi=300, bbox_inches="tight")
plt.close()
print("\n已保存 fig/fig4_triple_box.png")