# -*- coding: utf-8 -*-
"""卡 R1 的 1.5 万段档实测：dev 200 题，用原问题整句检索，
统计 gold 段落「至少一篇 / 两篇都」进 top-k 的命中率（k=1/3/5/10）。

bge 规范：query 侧加官方检索指令前缀，passage 侧不加。
"""
import json, os, torch
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
Q_PREFIX = "Represent this sentence for searching relevant passages: "

corpus = [json.loads(l) for l in open(f"{ROOT}/data/corpus.jsonl")]
V = torch.load(f"{ROOT}/data/corpus_vecs.pt")               # (N,384) 已归一化
titles = [c["title"] for c in corpus]

qs = [json.loads(l) for l in open(f"{ROOT}/data/subset_dev.jsonl")]

tok = AutoTokenizer.from_pretrained(os.environ.get("BGE_MODEL", "BAAI/bge-small-en-v1.5"))
model = AutoModel.from_pretrained(os.environ.get("BGE_MODEL", "BAAI/bge-small-en-v1.5")).to(DEV).eval()

with torch.no_grad():
    enc = tok([Q_PREFIX + e["question"] for e in qs], padding=True, truncation=True,
              max_length=128, return_tensors="pt").to(DEV)
    Q = torch.nn.functional.normalize(model(**enc).last_hidden_state[:, 0], dim=-1).cpu()

sims = Q @ V.T                                               # (200, N)
KS = (1, 3, 5, 10)
rank = sims.argsort(dim=1, descending=True)[:, :max(KS)]

stats = {k: {"at_least_one": 0, "both": 0} for k in KS}
for i, e in enumerate(qs):
    gold = set(e["supporting_facts"]["title"])
    got = [titles[j] for j in rank[i].tolist()]
    for k in KS:
        hit = gold & set(got[:k])
        stats[k]["at_least_one"] += bool(hit)
        stats[k]["both"] += len(hit) == 2

n = len(qs)
print(f"dev {n} 题 × 库 {len(corpus)} 段（问题整句做 query）")
for k in KS:
    s = stats[k]
    print(f"  top-{k:<2}  至少一篇 gold: {s['at_least_one']/n:6.1%}   两篇都在: {s['both']/n:6.1%}")
