"""合成协议演示，不加载模型，不代表真实评测成绩。"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from em import em

query = re.search(r'<search>(.*?)</search>', '<search>Demo Labs headquarters</search>').group(1)
observation = '<information>Demo Labs has its headquarters in Example City.</information>'
model_text = '<answer>Example City</answer>'
pred = re.search(r'<answer>(.*?)</answer>', model_text).group(1)
assert em(pred, 'Example City') == 1
assert em('Example City, Example Country', 'Example City') == 0
print(json.dumps({'synthetic': True, 'query': query, 'observation': observation,
                  'generated_text': model_text, 'prediction': pred,
                  'fixture_EM': em(pred, 'Example City')}, ensure_ascii=False, indent=2))
