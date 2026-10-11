# -*- coding: utf-8 -*-
"""建检索库：把 train+dev 全部题目的段落按 title 去重，
整篇入库（标题 + 句子拼接，「五个决策」第 3 条），
用 bge-small-en-v1.5 编码，存 corpus.jsonl + corpus_vecs.pt。

编码方式照搬法律项目 rag_v3_cite.py：CLS pooling + L2 归一化，
检索用矩阵乘法（库仅 1.5 万段，暴力精确检索，不用 FAISS——量级判断见调研文档）。
"""
import json, os, time, torch
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 128

# —— 收集段落，按 title 去重（同名文章取首次出现的版本）——
seen, corpus = set(), []
for split in ("subset_train", "subset_dev"):
    with open(f"{ROOT}/data/{split}.jsonl") as f:
        for line in f:
            e = json.loads(line)
            for t, ss in zip(e["context"]["title"], e["context"]["sentences"]):
                if t in seen:
                    continue
                seen.add(t)
                corpus.append({"pid": len(corpus), "title": t,
                               "text": t + ": " + " ".join(s.strip() for s in ss)})

with open(f"{ROOT}/data/corpus.jsonl", "w") as f:
    for c in corpus:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# —— 编码 ——
tok = AutoTokenizer.from_pretrained(os.environ.get("BGE_MODEL", "BAAI/bge-small-en-v1.5"))
model = AutoModel.from_pretrained(os.environ.get("BGE_MODEL", "BAAI/bge-small-en-v1.5")).to(DEV).eval()

vecs, t0 = [], time.time()
with torch.no_grad():
    for i in range(0, len(corpus), BATCH):
        batch = [c["text"] for c in corpus[i:i + BATCH]]
        enc = tok(batch, padding=True, truncation=True, max_length=256,
                  return_tensors="pt").to(DEV)
        out = model(**enc).last_hidden_state[:, 0]          # CLS pooling（bge 规范）
        vecs.append(torch.nn.functional.normalize(out, dim=-1).cpu())
        if (i // BATCH) % 20 == 0:
            print(f"  {i}/{len(corpus)}  {time.time()-t0:.0f}s")

V = torch.cat(vecs)                                          # (N, 384) float32
torch.save(V, f"{ROOT}/data/corpus_vecs.pt")
print(f"段落数(去重后): {len(corpus)}   向量: {tuple(V.shape)} "
      f"{V.element_size()*V.nelement()/1e6:.1f}MB   编码耗时: {time.time()-t0:.0f}s   设备: {DEV}")
