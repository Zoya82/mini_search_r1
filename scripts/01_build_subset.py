# -*- coding: utf-8 -*-
"""抽子集：执行「五个决策」的第 1/2/4 条。

筛选规则（见 实操.md「五个决策」）：
  type == bridge          # comparison 的 yes/no 会奖励瞎猜
  level == medium         # hard 可能全零，easy 可能没区分度
  len(context.title)==10  # 过滤 262 条零干扰异常样本

发现（2026-08-10）：官方 validation 全部是 hard，没有 medium。
→ 决策变更：评测集不从 validation 抽，从 train 的合格池里留出 200 题，
  与训练 1200 题不相交（同分布、不重叠）。
"""
import json, os, random

# 如需离线加载，请在运行前设置 HF_HUB_OFFLINE=1。
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
SEED = 42
N_TRAIN, N_DEV = 1200, 200

random.seed(SEED)
tr = load_dataset("hotpotqa/hotpot_qa", "distractor")["train"]

# —— 逐级过滤，每级记数（决策执行的证据）——
n_all = len(tr)
bm = [e for e in tr if e["type"] == "bridge" and e["level"] == "medium"]
n_bm = len(bm)
pool = [e for e in bm if len(e["context"]["title"]) == 10]
n_pool = len(pool)

# —— 抽样：先抽 1400，前 1200 训练、后 200 评测（不相交）——
picked = random.sample(pool, N_TRAIN + N_DEV)
train, dev = picked[:N_TRAIN], picked[N_TRAIN:]

def dump(rows, path):
    with open(path, "w") as f:
        for e in rows:
            f.write(json.dumps({
                "id": e["id"], "question": e["question"], "answer": e["answer"],
                "type": e["type"], "level": e["level"],
                "supporting_facts": e["supporting_facts"],
                "context": e["context"],
            }, ensure_ascii=False) + "\n")

dump(train, f"{ROOT}/data/subset_train.jsonl")
dump(dev, f"{ROOT}/data/subset_dev.jsonl")

print(f"train 全量           : {n_all}")
print(f"bridge+medium        : {n_bm}")
print(f"再滤 10 段整齐       : {n_pool}  (滤掉 {n_bm - n_pool} 条)")
print(f"抽出 train/dev       : {len(train)}/{len(dev)}  (seed={SEED}, 不相交)")
print("写入 data/subset_train.jsonl, data/subset_dev.jsonl")
