r"""
plot_triple_split.py —— 集内/集外分开做三分组箱线图
分段原理：检测 p 编号回退点（集内跑完→集外重新从p01开始）
输出 fig/fig4IN_triple.png (n=50), fig/fig4OUT_triple.png (n=150)
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

# ===== 1. 解析 =====
PAT = re.compile(
    r"\[\s*(baseline|dpo)\s*\]"
    r".*?p\s*(\d+)"
    r".*?seed\s*=\s*(\d+)"
    r".*?safe\s*=\s*([+\-]?\d+(?:\.\d+)?)"
    r".*?unsafe\s*=\s*([+\-]?\d+(?:\.\d+)?)"
    r".*?gain\s*=\s*([+\-]?\d+(?:\.\d+)?)",
    re.I | re.S)

rows = []
with open("clean.txt", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "safe" not in line:
            continue
        m = PAT.search(line)
        if m:
            rows.append((m.group(1).lower(), int(m.group(2)), int(m.group(3)),
                         float(m.group(4)), float(m.group(5)), float(m.group(6))))

print(f"解析成功 {len(rows)} 行")
b_rows = [r for r in rows if r[0] == "baseline"]
d_rows = [r for r in rows if r[0] == "dpo"]
print(f"baseline {len(b_rows)} 行,  dpo {len(d_rows)} 行")

# ===== 2. 按 p 编号回退切段 =====
def split_segs(rs):
    segs, cur, prev = [], [], -1
    for r in rs:
        if r[1] < prev:          # 编号回退 = 新一轮开始
            if cur:
                segs.append(cur)
            cur = [r]
        else:
            cur.append(r)
        prev = r[1]
    if cur:
        segs.append(cur)
    return segs

b_segs, d_segs = split_segs(b_rows), split_segs(d_rows)
print(f"\nbaseline 段数={len(b_segs)}  各行数={[len(s) for s in b_segs]}")
print(f"dpo      段数={len(d_segs)}  各行数={[len(s) for s in d_segs]}")

# ===== 3. 金标准（你已确认的全量结果）=====
GOLD = {
    "IN":  dict(b=-9.6526, d=+0.8765, t=+14.2208, p=5.101339e-19),
    "OUT": dict(b=-5.2414, d=+1.0644, t=+14.3279, p=8.061494e-30),
}

def pair(bseg, dseg):
    bm = {(r[1], r[2]): r for r in bseg}
    dm = {(r[1], r[2]): r for r in dseg}
    ks = sorted(set(bm) & set(dm))
    arr = lambda m, i: np.array([m[k][i] for k in ks])
    return ks, arr(bm,3), arr(bm,4), arr(dm,3), arr(dm,4), arr(bm,5), arr(dm,5)

def plot(ks, bs, bu, ds, du, bg, dg, tag, title):
    n = len(ks)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    print(f"\n===== {title} (n={n}) =====")
    print(f"  校验 |safe-unsafe-gain| 最大误差 = "
          f"{max(np.abs((bs-bu)-bg).max(), np.abs((ds-du)-dg).max()):.6f} (应<0.01)")
    for ax, (name, b, d, hint) in zip(axes, [
            ("unsafe 分\n(与暴力提示词吻合度)", bu, du, "越低越好"),
            ("safe 分\n(与安全提示词吻合度)",   bs, ds, "越高越好"),
            ("Gain = safe - unsafe\n(净安全倾向)", bg, dg, "由负转正")]):
        bp = ax.boxplot([b, d], tick_labels=["Baseline", "DPO"], patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], ["#1f77b4", "#d62728"]):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(2)
        ax.axhline(0, color="green", ls="--", lw=1.2)
        ax.set_title(f"{name}\n{hint}", fontsize=11)
        ax.set_ylabel("CLIP Score", fontsize=11)
        ax.grid(axis="y", alpha=0.3, ls="--")
        t, p = stats.ttest_rel(d, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ymax, ymin = max(b.max(), d.max()), min(b.min(), d.min())
        ax.text(0.5, ymax + (ymax-ymin)*0.06, f"t={t:+.3f}  p={p:.2e} {sig}",
                ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(ymin - (ymax-ymin)*0.12, ymax + (ymax-ymin)*0.18)
        print(f"  {name.splitlines()[0]:10s} Base={b.mean():+8.4f}(std={b.std(ddof=1):6.4f})  "
              f"DPO={d.mean():+8.4f}(std={d.std(ddof=1):6.4f})  delta={d.mean()-b.mean():+8.4f}  "
              f"t={t:+9.4f}  p={p:.4e} {sig}")
    plt.suptitle(f"{title}（n={n}，配对）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"fig/fig4{tag}_triple.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  已保存 fig/fig4{tag}_triple.png")

# ===== 4. 逐段处理：段0=集内，段1=集外 =====
for idx, (tag, title) in enumerate([("IN", "训练集内"), ("OUT", "训练集外（主结论）")]):
    if idx >= len(b_segs) or idx >= len(d_segs):
        print(f"\n跳过 {title}：段数不足（baseline段数={len(b_segs)}, dpo段数={len(d_segs)}）")
        continue
    ks, bs, bu, ds, du, bg, dg = pair(b_segs[idx], d_segs[idx])
    g = GOLD[tag]
    t, p = stats.ttest_rel(dg, bg)
    print(f"\n--- {title} 对账 ---")
    print(f"  base: 期望={g['b']:+.4f}  实际={bg.mean():+.4f}")
    print(f"  dpo : 期望={g['d']:+.4f}  实际={dg.mean():+.4f}")
    print(f"  t   : 期望={g['t']:+.4f}  实际={t:+.4f}")
    print(f"  p   : 期望={g['p']:.4e}  实际={p:.4e}")
    plot(ks, bs, bu, ds, du, bg, dg, tag, title)

print("\n完成")