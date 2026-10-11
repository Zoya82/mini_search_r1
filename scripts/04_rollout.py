# -*- coding: utf-8 -*-
"""站 3：单轮 rollout——一道题跑完整条轨迹并打印。

流程（对应流程图的格子）：
    问题进 → 生成到 </search> 停 → 解析 query → 检索 top-3 → 拼回 → 续写答案 → 提取答案 → EM

轨迹里三类 token：
    prompt token      = system 指令 + few-shot + 问题        （系统塞的，loss_mask=0）
    assistant token   = 模型两次生成的文本                    （模型写的，loss_mask=1）
    observation token = 检索回来的 top-3 段落                  （系统塞的，loss_mask=0）
"""
import json, os, re, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, StoppingCriteria, StoppingCriteriaList

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from em import em                                    # 站 2 写好的 EM 判分函数

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

# ============================================================
# 0. 加载：生成模型 + 检索器。都只加载一次，全局共用。
# ============================================================
MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()

# 检索器：与 03_hit_rate.py 完全一致
Q_PREFIX = "Represent this sentence for searching relevant passages: "     # bge 规范：query 侧加指令前缀
corpus = [json.loads(l) for l in open(f"{ROOT}/data/corpus.jsonl")]        # 13,652 段原文（书架）
V = torch.load(f"{ROOT}/data/corpus_vecs.pt")                              # (N,384) 已归一化的向量矩阵（卡片柜）
bge_tok = AutoTokenizer.from_pretrained(os.environ.get("BGE_MODEL", "BAAI/bge-small-en-v1.5"))
bge = AutoModel.from_pretrained(os.environ.get("BGE_MODEL", "BAAI/bge-small-en-v1.5")).to(DEV).eval()


# ============================================================
# 1. 生成：给对话历史，生成到 stop_text 出现为止（站 1 的代码包成函数）
# ============================================================
class StopOnText(StoppingCriteria):
    """generate 每吐一个 token 调一次 __call__；返回 True 停、False 继续。"""

    def __init__(self, tok, stop_text, prompt_len):
        self.tok = tok
        self.stop_text = stop_text
        self.prompt_len = prompt_len        # 只检查新生成的部分，避免 prompt 里 few-shot 的标签误触发

    def __call__(self, input_ids, scores, **kw):
        new_text = self.tok.decode(input_ids[0, self.prompt_len:])
        return self.stop_text in new_text   # 解码成文字再找子串：不管标签被切成几个 token 都能命中


def generate_until(messages, stop_text=None, max_new_tokens=128):
    """输入 messages 和停止标签，返回新生成的文本（不含 prompt）。stop_text=None 表示不设停止条件。"""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(DEV)
    prompt_len = enc.input_ids.shape[1]
    stop = StoppingCriteriaList([StopOnText(tok, stop_text, prompt_len)]) if stop_text else None
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             stopping_criteria=stop, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, prompt_len:], skip_special_tokens=True)


# ============================================================
# 2. 解析：从生成文本里取出 <search>…</search> 中间的 query（找不到返回 None → ABORTED）
# ============================================================
def parse_query(text):
    m = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ============================================================
# 3. 检索：query → top-k 段落原文（从 03_hit_rate.py 抽出）
# ============================================================
def retrieve(query, k=3):
    with torch.no_grad():
        enc = bge_tok([Q_PREFIX + query], padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(DEV)   # query 加前缀后编码
        q = torch.nn.functional.normalize(bge(**enc).last_hidden_state[:, 0], dim=-1).cpu()  # CLS 向量 + L2 归一化
    sims = (q @ V.T)[0]                                   # 与全库 N 段算余弦相似度 → (N,)
    idx = sims.argsort(descending=True)[:k].tolist()      # 取相似度最高的 k 个下标
    return [corpus[i]["text"] for i in idx]               # 回书架取原文


# ============================================================
# 5. 提取答案：从续写文本里取出 <answer>…</answer> 中间的内容
# ============================================================
def parse_answer(text):
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ============================================================
# prompt：系统指令 + 1 个 few-shot 例子（教格式）
# ============================================================
SYSTEM = (
    "You answer multi-hop questions with the help of a search tool.\n"
    "First write <search>your query</search> and stop.\n"
    "You will then receive search results inside <information>...</information>.\n"
    "Then write your final answer inside <answer>...</answer>. Keep the answer short (a few words)."
)
FEWSHOT = [
    {"role": "user", "content": "Question: In which city is the headquarters of the hotel company owned by the Oberoi family?"},
    {"role": "assistant", "content": "<search>Oberoi family hotel company</search>"},
    {"role": "user", "content": "<information>The Oberoi Group is a hotel company with its head office in Delhi.</information>"},
    {"role": "assistant", "content": "<answer>Delhi</answer>"},
]


# ============================================================
# 7. 串起来：一道题跑完整条单轮轨迹
# ============================================================
def run_one(question, gold, fake_search=None):
    """跑一道题。fake_search：SFT 前模型不吐 <search>，可手工塞一段含 <search> 的文本走通后半段。"""
    traj = {"question": question, "gold": gold, "status": None, "reward": 0}

    # —— 第一次生成：期望 <search>query</search> ——            [assistant token]
    messages = [{"role": "system", "content": SYSTEM}] + FEWSHOT + [
        {"role": "user", "content": f"Question: {question}"}]
    raw1 = fake_search if fake_search is not None else generate_until(messages, stop_text="</search>")
    traj["gen1"] = raw1

    query = parse_query(raw1)
    if query is None:                                     # 解析失败 → ABORTED（对应 rollout.py 的第三种出口）
        traj["status"] = "ABORTED"
        return traj
    traj["query"] = query

    # —— 检索 → observation ——                                [observation token]
    passages = retrieve(query, k=3)
    obs = "<information>" + "\n".join(passages) + "</information>"
    traj["obs"] = obs

    # —— 拼回：模型的话挂 assistant，环境的话挂 user ——
    messages += [{"role": "assistant", "content": raw1},
                 {"role": "user", "content": obs}]

    # —— 第二次生成：期望 <answer>...</answer> ——              [assistant token]
    raw2 = generate_until(messages, stop_text="</answer>")
    traj["gen2"] = raw2

    pred = parse_answer(raw2)
    traj["pred"] = pred
    traj["status"] = "COMPLETED" if pred is not None else "ABORTED"
    traj["reward"] = em(pred, gold) if pred is not None else 0   # 终止步才有非零 reward
    return traj


def print_traj(t):
    print("=" * 70)
    print(f"[PROMPT   ]  Question: {t['question']}")
    print(f"[ASSISTANT]  {t.get('gen1')!r}")
    if "obs" in t:
        print(f"[OBS      ]  {t['obs'][:300]!r}{' ...' if len(t['obs']) > 300 else ''}")
        print(f"[ASSISTANT]  {t.get('gen2')!r}")
    print(f"pred={t.get('pred')!r}   gold={t['gold']!r}   status={t['status']}   reward={t['reward']}")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10          # 跑几题：python3 04_rollout.py 20
    dev = [json.loads(l) for l in open(f"{ROOT}/data/subset_dev.jsonl")][:N]

    stats = {"has_search": 0, "has_answer": 0, "completed": 0, "em": 0}
    for e in dev:
        t = run_one(e["question"], e["answer"])
        print_traj(t)
        stats["has_search"] += t.get("query") is not None
        stats["has_answer"] += t.get("pred") is not None
        stats["completed"] += t["status"] == "COMPLETED"
        stats["em"] += t["reward"]

    print("\n" + "#" * 70)
    print(f"dev 前 {N} 题（SFT 前，few-shot 引导）：")
    print(f"  吐出 <search>       {stats['has_search']}/{N}")
    print(f"  吐出 <answer>       {stats['has_answer']}/{N}")
    print(f"  COMPLETED           {stats['completed']}/{N}")
    print(f"  EM=1                {stats['em']}/{N}")
