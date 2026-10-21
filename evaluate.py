import argparse
import json
from pathlib import Path

import yaml

from evaluators.math import evaluate as eval_math
from evaluators.medical import evaluate as eval_medical
from evaluators.medical_mcq_v1 import evaluate as eval_medical_mcq
from evaluators.summarization import evaluate as eval_summary

ROOT = Path(__file__).resolve().parent

EVALUATORS = {
    "math_reasoning": eval_math,
    "text_summarization": eval_summary,
    "medical_qa": eval_medical,
    "medical_mcq_xmedbench_v1": eval_medical_mcq,
}


def read_jsonl(path):
    """读取 JSONL 文件，并报告具体的错误行。"""
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_no} 行 JSON 格式错误：{exc}"
                ) from exc

    return records


def resolve_path(value, description):
    """将相对路径解析为项目根目录下的路径，并检查文件存在。"""
    if not value:
        raise ValueError(f"未提供{description}路径")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"{description}文件不存在：{path}")

    return path.resolve()


def load_dataset_path(task, explicit_data):
    """优先使用 --data；未指定时读取公共任务配置。"""
    if explicit_data:
        return resolve_path(explicit_data, "测试集")

    config_path = ROOT / "configs" / "benchmark.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Benchmark 配置文件不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    tasks = config.get("tasks", {})
    if task not in tasks:
        raise ValueError(f"Benchmark 配置中没有任务：{task}")

    task_cfg = tasks[task] or {}
    dataset_value = task_cfg.get("dataset_path") or task_cfg.get("dataset")

    if not dataset_value:
        raise ValueError(
            f"任务 {task} 未配置 dataset_path 或 dataset"
        )

    return resolve_path(dataset_value, "测试集")


def main():
    parser = argparse.ArgumentParser(
        description="使用指定任务的评估器，对 Benchmark 预测结果进行评分"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(EVALUATORS.keys()),
    )
    parser.add_argument(
        "--data",
        default=None,
        help="可选：显式指定测试集 JSONL；未指定时读取 configs/benchmark.yaml",
    )
    args = parser.parse_args()

    pred_path = resolve_path(args.predictions, "预测结果")
    data_path = load_dataset_path(args.task, args.data)

    predictions = read_jsonl(pred_path)
    dataset = read_jsonl(data_path)

    references = {}
    for item in dataset:
        if "id" not in item or "reference" not in item:
            raise ValueError(
                f"测试集记录缺少 id 或 reference 字段：{item}"
            )
        sample_id = item["id"]
        if sample_id in references:
            raise ValueError(f"测试集存在重复 id：{sample_id}")
        references[sample_id] = {
            "reference": item["reference"],
            "metadata": item.get("metadata", {}),
        }

    for item in predictions:
        if "sample_id" not in item:
            raise ValueError(f"预测记录缺少 sample_id：{item}")

        sample_id = item["sample_id"]
        if sample_id not in references:
            raise ValueError(
                f"预测中出现测试集不存在的 sample_id：{sample_id}"
            )

        matched = references[sample_id]
        item["reference"] = matched["reference"]
        item["metadata"] = matched["metadata"]

    metrics = EVALUATORS[args.task](predictions)
    metrics["task"] = args.task
    metrics["predictions_file"] = str(pred_path)
    metrics["dataset_file"] = str(data_path)

    output_path = pred_path.parent / "metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"已保存：{output_path}")


if __name__ == "__main__":
    main()
