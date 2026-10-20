import argparse
import json
from pathlib import Path
import yaml

from evaluators.math import evaluate as eval_math
from evaluators.medical import evaluate as eval_medical
from evaluators.summarization import evaluate as eval_summary

ROOT = Path(__file__).resolve().parent


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--task", required=True,
                        choices=["medical_qa", "math_reasoning", "text_summarization"])
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = ROOT / pred_path

    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
    else:
        cfg = yaml.safe_load((ROOT / "configs/benchmark.yaml").read_text(encoding="utf-8"))
        data_path = ROOT / cfg["tasks"][args.task]["dataset"]

    predictions = read_jsonl(pred_path)
    references = {x["id"]: x["reference"] for x in read_jsonl(data_path)}
    for item in predictions:
        sample_id = item["sample_id"]
        if sample_id not in references:
            raise ValueError(f"预测中出现测试集不存在的 sample_id: {sample_id}")
        item["reference"] = references[sample_id]

    evaluator = {
        "math_reasoning": eval_math,
        "text_summarization": eval_summary,
        "medical_qa": eval_medical,
    }[args.task]
    metrics = evaluator(predictions)
    metrics["task"] = args.task
    metrics["predictions_file"] = str(pred_path)
    output_path = pred_path.parent / "metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"已保存：{output_path}")


if __name__ == "__main__":
    main()
