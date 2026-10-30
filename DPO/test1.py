import numpy as np
from scipy import stats
d = {ep: np.load(f"s_dpo_out_epoch_{ep}.npy") for ep in [10,30,50]}
for a,b in [(30,10),(50,10),(30,50)]:
    x,y = d[a], d[b]
    t,p = stats.ttest_rel(x,y)
    diff = x-y
    print(f"epoch_{a} vs epoch_{b}: Δ={diff.mean():+.4f}  t={t:+.4f}  df=149  p={p:.4f}")
print("\nBonferroni 校正阈值 α' = 0.05/3 =", 0.05/3)