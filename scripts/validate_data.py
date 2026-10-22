import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REQUIRED = {"id", "task", "input", "reference"}


def main():
    # 同时覆盖固定名 test.jsonl、带版本后缀 test_*.jsonl
    # 与 *_v1.jsonl（如 lcsts_test_v1、xmedbench_zh_v1）。
    patterns = ("*/test.jsonl", "*/test_*.jsonl", "*/*_v1.jsonl")
    files = sorted(
        {path for pattern in patterns for path in DATA.glob(pattern)}
    )
    if not files:
        raise SystemExit("未找到 data/*/test.jsonl")
    for path in files:
        seen = set()
        count = 0
        expected_task = path.parent.name
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}:{line_no}: JSON 错误: {e}")
                missing = REQUIRED - obj.keys()
                if missing:
                    raise SystemExit(f"{path}:{line_no}: 缺字段 {missing}")
                if not all(isinstance(obj[k], str) and obj[k].strip()
                           for k in ("id", "task", "input")):
                    raise SystemExit(f"{path}:{line_no}: id/task/input 必须非空字符串")
                if not isinstance(obj["reference"], str):
                    raise SystemExit(f"{path}:{line_no}: reference 必须是字符串")
                if obj["task"] != expected_task:
                    raise SystemExit(
                        f"{path}:{line_no}: task={obj['task']}，应为 {expected_task}"
                    )
                if obj["id"] in seen:
                    raise SystemExit(f"{path}:{line_no}: ID 重复 {obj['id']}")
                seen.add(obj["id"])
                count += 1
        print(f"[OK] {path.relative_to(ROOT)}: {count} 条")
    print("数据格式检查通过。")


if __name__ == "__main__":
    main()
