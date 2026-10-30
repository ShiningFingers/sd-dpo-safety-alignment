import numpy as np
from scipy import stats

print("=== 训练轮数消融：配对 t 检验（集外 n=150）===\n")
b = np.load("s_base_out.npy")
print(f"Baseline : n={len(b)}  mean={b.mean():+.4f}  std={b.std(ddof=1):.4f}\n")

for ep in [10, 30, 50]:
    d = np.load(f"s_dpo_out_epoch_{ep}.npy")
    assert len(d) == len(b), f"epoch_{ep} 样本数不匹配!"
    diff = d - b
    n = len(diff)
    t, p = stats.ttest_rel(d, b)
    se = diff.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    coh = diff.mean() / diff.std(ddof=1)
    print(f"epoch_{ep:<3}: mean={d.mean():+.4f}  std={d.std(ddof=1):.4f}")
    print(f"          Δ={diff.mean():+.4f}  SE={se:.4f}")
    print(f"          t={t:+.4f}  df={n-1}  p={p:.3e}")
    print(f"          Cohen's d={coh:+.4f}  95%CI=[{diff.mean()-tc*se:+.4f}, {diff.mean()+tc*se:+.4f}]\n")

d30 = np.load("s_dpo_out_epoch_30.npy")
d50 = np.load("s_dpo_out_epoch_50.npy")
diff = d30 - d50
t, p = stats.ttest_rel(d30, d50)
print(f"=== epoch_30 vs epoch_50（两 checkpoint 互相比对）===")
print(f"Δ={diff.mean():+.4f}  t={t:+.4f}  df={len(diff)-1}  p={p:.4f}")
print("p<0.05 → 30 轮确实显著优于 50 轮；p>0.05 → 只能说'观察到下降趋势'")