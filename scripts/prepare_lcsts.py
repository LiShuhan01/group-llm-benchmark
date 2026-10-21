"""从本地 LCSTS 数据集中生成固定随机种子的摘要评估集。"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/data/lsh/datasets/LCSTS",
        help="load_from_disk 保存的 LCSTS 数据集目录",
    )
    parser.add_argument(
        "--output",
        default="data/text_summarization/lcsts_test_v1.jsonl",
        help="输出 JSONL 文件路径",
    )
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = load_from_disk(args.source)
    if "validation" not in source:
        raise ValueError("数据集中没有 validation 分割")

    validation = source["validation"]
    valid_indices = [
        i for i, row in enumerate(validation)
        if str(row.get("text") or "").strip()
        and str(row.get("summary") or "").strip()
    ]

    if args.size <= 0 or args.size > len(valid_indices):
        raise ValueError(
            f"请求样本数 {args.size} 不合法；"
            f"可用样本数为 {len(valid_indices)}"
        )

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(valid_indices, args.size))

    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for i in selected:
            row = validation[i]
            record = {
                "id": f"lcsts_val_{i:06d}",
                "task": "text_summarization",
                "input": row["text"].strip(),
                "reference": row["summary"].strip(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"输出文件：{output}")
    print(f"样本数量：{len(selected)}")
    print(f"随机种子：{args.seed}")


if __name__ == "__main__":
    main()
