r"""
ablation.py —— 训练轮数消融（只跑 DPO 组，baseline 复用）
产出：s_dpo_out_epoch_10.npy, s_dpo_out_epoch_30.npy
      epoch_50 直接复用 s_dpo_out.npy（因为 LORA_PATH 本就指向它）
"""
import os, shutil
import numpy as np
import torch
import eval_dpo

print("=== 训练轮数消融：只跑 DPO 组 ===")

# 1. 复用 baseline（原始 SD 与训练轮数无关）
b_out = np.load("s_base_out.npy")
print(f"[复用] baseline 集外 n={len(b_out)}  mean={b_out.mean():+.4f}  (应=-5.2414)")
print(f"[配置] PAIRS_OUT 词数={len(eval_dpo.PAIRS_OUT)}  SEEDS={eval_dpo.SEEDS}")

# 想多加就往这里加，每加一个约 25 分钟
CKPTS = ["epoch_10", "epoch_30"]

for ck in CKPTS:
    path = f"./lora_safety_aligned_{ck}"
    if not os.path.exists(path):
        print(f"[跳过] 找不到 {path}")
        continue
    eval_dpo.LORA_PATH = path          # ← 核心：只换路径
    print(f"\n=== {ck} : {path} ===")
    pipe = eval_dpo.load_pipeline(use_lora=True)
    d_out = eval_dpo.generate_and_score(pipe, eval_dpo.PAIRS_OUT, f"dpo-{ck}")
    np.save(f"s_dpo_out_{ck}.npy", d_out)
    print(f"{ck} 完成: gain mean = {d_out.mean():+.4f}  n={len(d_out)}")
    del pipe
    torch.cuda.empty_cache()

# 3. epoch_50 复用已有的
if os.path.exists("s_dpo_out.npy") and not os.path.exists("s_dpo_out_epoch_50.npy"):
    shutil.copy("s_dpo_out.npy", "s_dpo_out_epoch_50.npy")
d50 = np.load("s_dpo_out_epoch_50.npy")
print(f"\n[复用] epoch_50 gain mean = {d50.mean():+.4f}  (应=+1.0644)")

print("\n完成。接着跑: python plot_ablation.py")