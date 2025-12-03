import json
import sys
import os

def sorted_dict(d, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = {"STAT_DESC", "performance", "memory_type"}  # 需要忽略的新增字段
    if isinstance(d, dict):
        # 过滤掉需要忽略的键
        filtered = {k: sorted_dict(v, ignore_keys) for k, v in sorted(d.items()) if k not in ignore_keys}
        return filtered
    elif isinstance(d, list):
        sorted_items = [sorted_dict(item, ignore_keys) for item in d]
        return sorted(sorted_items, key=lambda x: x.get("PC", 0) if isinstance(x, dict) else 0)
    else:
        return d

# 从命令行获取prog名称（如"prog1"）
if len(sys.argv) < 2:
    sys.exit(1)

prog_name = sys.argv[1]
output_filename = f"output_opt.json"
answer_filename = f"answer/{prog_name}.json"

# 读取output(utf-16 le)
try:
    with open(output_filename, 'r', encoding='utf-16') as f:
        output = json.load(f)
        output = output.get("trace", [])
except (FileNotFoundError, json.JSONDecodeError):
    print("false")
    sys.exit(1)

# 读取answer(utf-8)
try:
    with open(answer_filename, 'r', encoding='utf-8') as f:
        answer = json.load(f)
        if not isinstance(answer, list):
            answer = answer.get("trace", [])
except (FileNotFoundError, json.JSONDecodeError):
    print("false")
    sys.exit(1)

# 排序比较
sorted_output = sorted_dict(output)
sorted_answer = sorted_dict(answer)

print("true" if sorted_output == sorted_answer else "false")