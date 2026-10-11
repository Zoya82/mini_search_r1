# -*- coding: utf-8 -*-
"""玩具示例：让 model.generate 在出现指定文字时立刻停下。

任务：让模型自我介绍，但我们要求它一说完第一句话（出现第一个句号）就停。
你的任务：跑通并读懂这个文件，然后合上它，在 04_rollout.py 里写出
        "生成到 </search> 就停"的版本。停的目标不同，机制完全一样。

运行：python3 scripts/practice/stop_demo.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# 模型借用法律项目里现成的 Qwen2.5-0.5B-Instruct，不用下载
MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()


class StopOnText(StoppingCriteria):
    """核心：生成是一个 token 一个 token 吐的。每吐出一个新 token，
    generate 就调用一次这个类的 __call__ 问："该停了吗？"
    返回 True = 立刻停；返回 False = 继续吐下一个 token。
    """

    def __init__(self, tok, stop_text, prompt_len):
        self.tok = tok
        self.stop_text = stop_text      # 出现这段文字就停（本例是句号"。"）
        self.prompt_len = prompt_len    # prompt 占了多少个 token——只检查新生成的部分

    def __call__(self, input_ids, scores, **kw):
        # input_ids 是"prompt + 已生成的所有 token"，切掉前 prompt_len 个，
        # 剩下的才是模型新写的字
        new_tokens = input_ids[0, self.prompt_len:]
        new_text = self.tok.decode(new_tokens)
        return self.stop_text in new_text
        # 为什么用"解码成文字再找子串"，而不是直接比对 token 编号？
        # 因为同一段文字可能被切成不同的 token 组合（"</search>" 可能是
        # 3 个 token 也可能是 4 个），按编号比对会漏；解码成文字最稳。
        # 代价是每吐一个 token 就解码一次，慢一点——我们这个规模无所谓。


# —— 1. 拼 prompt（Qwen 是聊天模型，要走它的聊天模板）——
messages = [{"role": "user", "content": "Introduce yourself in English."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tok(prompt, return_tensors="pt").to(DEV)
prompt_len = enc.input_ids.shape[1]     # prompt 的 token 数，传给停止条件

# —— 2. 生成，挂上停止条件 ——
stop = StoppingCriteriaList([StopOnText(tok, ".", prompt_len)])
with torch.no_grad():
    out = model.generate(
        **enc,
        max_new_tokens=100,             # 保险丝：就算一直不出现句号，最多也只写 100 个 token
        do_sample=False,                # 不采样，同样输入永远同样输出，方便调试
        stopping_criteria=stop,
    )

# —— 3. 只解码新生成的部分 ——
new_text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
print("模型输出（应停在第一个句号处）：")
print(repr(new_text))
