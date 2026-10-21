import json
import re
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/medical_qa/xmedbench_zh_v1.jsonl"
DATASET_NAME = "FreedomIntelligence/XMedbench"
DATASET_CONFIG = "test"
SPLIT = "chinese"


def normalize_answer(answer):
    """将 (C)、(ACE) 等标准答案规范化为 C、ACE。"""
    answer = str(answer or "").strip().upper()
    match = re.fullmatch(r"\(?\s*([A-E]+)\s*\)?", answer)

    if not match:
        raise ValueError(f"无法识别标准答案格式：{answer!r}")

    letters = match.group(1)

    if len(set(letters)) != len(letters):
        raise ValueError(f"答案中存在重复选项字母：{answer!r}")

    return "".join(sorted(letters))


def main():
    print(f"加载数据集：{DATASET_NAME}，配置：{DATASET_CONFIG}")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    if SPLIT not in dataset:
        raise ValueError(f"缺少数据分割：{SPLIT}")

    data = dataset[SPLIT]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    invalid = 0

    with OUTPUT.open("w", encoding="utf-8") as f:
        for index, row in enumerate(data, start=1):
            question = str(row.get("question") or "").strip()
            options = str(row.get("options") or "").strip()
            source = str(row.get("source") or "").strip()

            if not question or not options:
                invalid += 1
                continue

            try:
                answer = normalize_answer(row.get("answer"))
            except ValueError as exc:
                raise ValueError(
                    f"第 {index} 条数据的标准答案异常：{exc}"
                ) from exc

            # 只有选项中实际存在的字母才允许作为标准答案。
            option_letters = set(
                re.findall(r"(?m)^\s*\(([A-E])\)", options)
            )
            if not option_letters or not set(answer) <= option_letters:
                raise ValueError(
                    f"第 {index} 条数据的答案与选项不匹配："
                    f"answer={answer}, options={options!r}"
                )

            sample_id = f"xmedbench_zh_{index:06d}"
            if sample_id in seen_ids:
                raise ValueError(f"重复 ID：{sample_id}")
            seen_ids.add(sample_id)

            prompt_input = (
                "请回答以下医学选择题。根据题目和选项选择正确答案。\n"
                "单选题请输出一个选项字母；多选题请输出所有正确选项字母，"
                "按字母顺序排列。最终答案请写成“答案：C”或“答案：ACE”，"
                "不要只给出解释而不明确作答。\n\n"
                f"问题：\n{question}\n\n"
                f"选项：\n{options}\n\n"
                "答案："
            )

            record = {
                "id": sample_id,
                "task": "medical_qa",
                "input": prompt_input,
                "reference": answer,
                "metadata": {
                    "dataset": DATASET_NAME,
                    "dataset_config": DATASET_CONFIG,
                    "split": SPLIT,
                    "source": source,
                    "source_index": index - 1,
                },
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"输出文件：{OUTPUT}")
    print(f"原始样本数：{len(data)}")
    print(f"跳过空问题或空选项的样本数：{invalid}")
    print(f"输出样本数：{len(seen_ids)}")


if __name__ == "__main__":
    main()
