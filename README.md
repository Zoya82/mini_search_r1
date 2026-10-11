# mini Search-R1：单轮检索 Agent 原型

一个学习型原型：把 HotpotQA 的数据准备、向量检索、标签协议、停止生成、工具观察回填和 EM 判分串成单轮 rollout。更新至 2026-10-11，当前已保存 SFT 前基线；冷启动数据构造仍在开发，尚无 Search-R1 强化学习训练结果。

## 已完成与已观察

- 从 HotpotQA train 的 bridge、medium、10段候选池中留出 1200／200 条训练与开发问答，seed＝42，两侧 ID 不重叠。
- 以 train＋dev 候选段落构造缩小检索库，按标题去重得到 13652 段；这是受限语料原型，不等同于 fullwiki。
- bge-small-en-v1.5：CLS pooling、L2 归一化、矩阵乘法精确检索。
- Qwen2.5-0.5B-Instruct：生成 `<search>` → 检索 top3 → 回填 `<information>` → 生成 `<answer>` → EM。
- [保存的前10题轨迹](outputs/rollout_dev10.txt)：search 标签 10／10，answer 标签和完成轨迹 3／10，严格 EM 0／10。只是这10题的观察，不外推到全部 Dev。

`scripts/05_make_sft_data.py` 保留 TODO，属于待实现的冷启动练习；当前没有完成 assistant-only loss masking、SFT 训练或 RL 更新。

## 运行

```bash
# 无模型、无网络，查看协议与 EM 判分的合成演示。
python3 examples/protocol_demo.py
python3 scripts/test_em.py

# 真实模型流程：需联网下载公开数据及模型。
python3 -m pip install -r requirements.txt
python3 scripts/01_build_subset.py
python3 scripts/02_build_corpus.py
python3 scripts/03_hit_rate.py
python3 scripts/04_rollout.py 10
```

模型默认使用 Hugging Face 标识，可用 `QWEN_MODEL`、`BGE_MODEL` 指向本地模型。发布副本修正了个人绝对路径、数据目录初始化及强制离线设置；原始实验文件保持原样。本次只运行协议演示和现有 EM 检查，没有重新下载模型或生成10题轨迹。

原始 HotpotQA、模型权重和向量缓存不分发，代码与数据来源分别遵循各自权利说明。研究背景为 Search-R1；本仓库展示自己当前完成的原型范围。
