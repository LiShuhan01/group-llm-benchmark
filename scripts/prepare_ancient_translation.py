"""Prepare classical translation JSONL for Group LLM Benchmark."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="原始 JSONL，至少包含 id/input/reference")
    parser.add_argument(
        "--output",
        default="data/ancient_translation/test_v1.jsonl",
        help="输出路径",
    )
    parser.add_argument("--source-name", default="user_provided_wywmt_test")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    count = 0
    with source.open("r", encoding="utf-8") as fin, output.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source} 第 {line_no} 行 JSON 格式错误: {exc}") from exc

            sample_id = str(row.get("id", "")).strip()
            input_text = str(row.get("input", "")).strip()
            reference = str(row.get("reference", row.get("output", ""))).strip()
            if not sample_id or not input_text or not reference:
                raise ValueError(f"第 {line_no} 行缺少 id、input 或 reference/output")
            if sample_id in seen:
                raise ValueError(f"重复 id: {sample_id}")
            seen.add(sample_id)

            item = {
                "id": sample_id,
                "task": "ancient_translation",
                "input": input_text,
                "reference": reference,
                "metadata": {
                    "source": args.source_name,
                    "split": "test",
                    "original_task": row.get("task", "ancient_translation"),
                },
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    print(f"输出文件：{output.resolve()}")
    print(f"样本数量：{count}")


if __name__ == "__main__":
    main()
