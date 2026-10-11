# -*- coding: utf-8 -*-
"""造冷启动 SFT 数据：拒绝采样（rejection sampling）为主，模板兜底。

流程：
    1. 用 04_rollout.py 的 run_one 跑 subset_train 前 N 题（模型自己生成 search 和 answer）
    2. 判断每条轨迹"答对了没"——标准是 gold 是否出现在第二次生成的文本里（不看有没有 <answer> 标签）
    3. 答对的轨迹 → 拼成 SFT 样本（messages 格式，和法律项目 sft_train.json 一样）
    4. 不够 MIN_SAMPLES 条 → 用 gold 段落 + gold 答案按同样格式拼模板样本补齐
    5. 存 data/sft_train.json

SFT 样本的 messages 格式（一条样本 = 一次完整单轮对话，5 条消息）：
    system    : SYSTEM 指令
    user      : Question: ...
    assistant : <search>query</search>               ← 算 loss
    user      : <information>...</information>        ← 不算 loss
    assistant : <answer>gold</answer>                  ← 算 loss
注意：few-shot 例子不进训练数据——SFT 就是为了让模型不靠 few-shot 也会这个格式。

用法：
    python3 scripts/05_make_sft_data.py          # 默认跑 train 前 200 题
    python3 scripts/05_make_sft_data.py 20       # 先跑 20 题试试
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 从 04_rollout.py 直接拿：run_one、SYSTEM、em、normalize、ROOT
# （04_rollout.py 顶部加载了两个模型，import 时会加载一次，正常）
from importlib import import_module
r04 = import_module("04_rollout")
run_one, SYSTEM, ROOT = r04.run_one, r04.SYSTEM, r04.ROOT
from em import em, normalize

N_RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 200   # 跑多少题
MIN_SAMPLES = 50                                          # 至少要凑够多少条


def is_correct(traj):
    """判断一条轨迹是否答对。

    输入：run_one 返回的字典（有 gen2、gold；ABORTED 时可能没有 gen2）
    输出：True / False

    标准：gold 归一化后，出现在 gen2 归一化后的文本里（子串匹配）。
    为什么不用 traj["reward"]：reward 依赖 <answer> 标签，会把"答对但没写标签"的漏掉，
    而那正是我们最想要的正样本（dev 10 题里有 2 题是这种）。
    提示：normalize 已经 import 进来了；gen2 可能不存在（ABORTED），要先判断。
    """
    # TODO


def traj_to_sample(traj):
    """把一条答对的轨迹拼成 SFT 样本。

    输入：run_one 返回的字典（question、query、obs、gold 都在里面）
    输出：{"messages": [5 条消息]}，格式见文件顶部 docstring

    注意两点：
    - 第一条 assistant 的内容直接用 "<search>" + query + "</search>" 重新拼，不用 gen1 原文
      （gen1 里可能有多余的字；重新拼保证格式干净）
    - 第二条 assistant 的内容用 "<answer>" + gold + "</answer>"——用 gold 不用 gen2：
      gen2 可能是一整句话，SFT 要教的是"只输出标签包着的短答案"
    """
    # TODO


def template_sample(example):
    """模板兜底：不跑模型，直接用数据集自带的 gold 段落拼一条样本。

    输入：subset_train.jsonl 的一条原始数据（有 question、answer、supporting_facts、context）
    输出：{"messages": [5 条消息]}，格式同上

    怎么拼：
    - query：直接用 question 本身（没有模型生成的 query，用原问题代替）
    - information：从 context 里找出 supporting_facts["title"] 对应的那几篇文章，
      每篇拼成 "标题: 句1 句2 …"（和建库时 corpus 的 text 格式一样），用 "\\n" 连接
      提示：context 是 {"title": [...], "sentences": [[...], ...]}，title[i] 对应 sentences[i]
    - answer：example["answer"]
    """
    # TODO


def main():
    train = [json.loads(l) for l in open(f"{ROOT}/data/subset_train.jsonl")]

    # —— 1. 拒绝采样：跑前 N_RUN 题，留答对的 ——
    samples, used_ids = [], set()
    stats = {"run": 0, "has_search": 0, "correct": 0}
    for e in train[:N_RUN]:
        # TODO：调 run_one，统计 stats（跑了几题、几题有 query、几题 is_correct）
        #       is_correct 的 → traj_to_sample 加进 samples，e["id"] 记进 used_ids
        pass
    print(f"拒绝采样：跑 {stats['run']} 题，有 search {stats['has_search']}，答对 {stats['correct']} → {len(samples)} 条样本")

    # —— 2. 不够就用模板补：从没用过的题里按顺序取，补到 MIN_SAMPLES ——
    n_template = 0
    for e in train:
        if len(samples) >= MIN_SAMPLES:
            break
        # TODO：跳过 used_ids 里的题；template_sample(e) 加进 samples；n_template += 1
    print(f"模板补充：{n_template} 条 → 共 {len(samples)} 条")

    # —— 3. 存盘 ——
    out = f"{ROOT}/data/sft_train.json"
    json.dump(samples, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"已存 {out}")
    print("第 1 条样本：")
    print(json.dumps(samples[0], ensure_ascii=False, indent=1)[:800])


if __name__ == "__main__":
    main()
