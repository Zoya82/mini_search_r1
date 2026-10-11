# -*- coding: utf-8 -*-
"""站 2 的验收文件：你在 scripts/em.py 里写 normalize() 和 em()，
然后运行 python3 scripts/test_em.py，8 个用例全过算通关。

规则（HotpotQA 官方评测的归一化标准）：
  normalize(text)：转小写 → 去标点 → 去冠词 a/an/the（作为独立单词时）→ 多个空白压成一个
  em(pred, gold)：normalize 后完全相等返回 1，否则返回 0
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from em import normalize, em   # ← 你要写的文件：scripts/em.py

CASES = [
    # (模型答案, 标准答案, 期望得分, 考察点)
    ("Delhi",              "Delhi",              1, "完全一致"),
    ("delhi",              "Delhi",              1, "大小写"),
    ("The Oberoi Group",   "Oberoi Group",       1, "去冠词 the"),
    ("Arthur's Magazine",  "arthurs magazine",   1, "去标点（撇号）"),
    (" 2017 ",             "2017",               1, "首尾空白"),
    ("No.",                "no",                 1, "标点加大小写"),
    ("Delhi, India",       "Delhi",              0, "多说了内容就是不对——EM 是全等"),
    ("a theater manager",  "theater manager",    1, "去冠词 a，且中间空白压平"),
]

fails = 0
for pred, gold, want, why in CASES:
    got = em(pred, gold)
    mark = "✅" if got == want else "❌"
    if got != want:
        fails += 1
        print(f"{mark} em({pred!r}, {gold!r}) = {got}，期望 {want}   ← {why}")
        print(f"     normalize(pred) = {normalize(pred)!r}")
        print(f"     normalize(gold) = {normalize(gold)!r}")
    else:
        print(f"{mark} em({pred!r}, {gold!r}) = {got}   （{why}）")

print()
print("全部通过 🎉" if fails == 0 else f"还有 {fails} 个用例没过，看上面的 normalize 输出找差异")

if fails:
    raise SystemExit(1)
