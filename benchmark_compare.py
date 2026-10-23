#!/usr/bin/env python3
"""一键比较基座模型与微调模型在一个或多个领域的表现。"""

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "outputs" / "runs"
REPORT_ROOT = ROOT / "outputs" / "comparisons"

COUNT_KEYS = {
    "num_samples", "num_scored", "num_failed",
    "correct", "invalid_count", "single_choice_count",
    "multiple_choice_count",
}


def read_yaml(path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configuration():
    benchmark = read_yaml(ROOT / "configs" / "benchmark.yaml")
    models = read_yaml(ROOT / "configs" / "models.yaml")
    return benchmark.get("tasks", {}), models.get("models", {})


def resolve_tasks(args, task_cfg, models):
    if args.base_model not in models:
        raise ValueError(f"模型配置不存在：{args.base_model}")
    if args.finetuned_model not in models:
        raise ValueError(f"模型配置不存在：{args.finetuned_model}")

    base_tasks = set(models[args.base_model].get("tasks", []))
    ft_tasks = set(models[args.finetuned_model].get("tasks", []))
    common = base_tasks & ft_tasks

    if args.scope == "domain":
        if args.task:
            selected = [args.task]
        elif len(common) == 1:
            selected = list(common)
        else:
            raise ValueError(
                "请使用 --task 指定目标领域；"
                "两个模型共同支持的任务不唯一。"
            )
    else:
        if args.task:
            raise ValueError("--scope all 不应同时指定 --task")
        selected = sorted(common)

    enabled = [
        t for t in selected
        if t in task_cfg and task_cfg[t].get("enabled", False)
    ]

    excluded = [t for t in selected if t not in enabled]
    if excluded:
        print("以下任务未启用或未配置，将不运行：", ", ".join(excluded))

    if not enabled:
        raise ValueError(
            "没有可运行的共同任务。请检查模型 tasks 和 "
            "configs/benchmark.yaml 中的 enabled 设置。"
        )

    return enabled


def run_command(command):
    """实时显示子进程输出，并在失败时中止当前评估。"""
    print("\n$", " ".join(map(str, command)), flush=True)
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)

    code = proc.wait()
    if code != 0:
        raise RuntimeError(
            f"命令执行失败，退出码 {code}：{' '.join(command)}"
        )
    return "".join(lines)


def run_one(model, task, limit):
    command = [
        sys.executable, "run_benchmark.py",
        "--model", model,
        "--task", task,
    ]
    if limit is not None:
        command += ["--limit", str(limit)]

    output = run_command(command)

    match = re.search(r"预测完成：\s*(.+)", output)
    if match:
        run_dir = Path(match.group(1).strip())
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
    else:
        # 兼容未打印“预测完成”的旧版脚本：
        # 从本次运行新产生的目录中定位结果。
        raise RuntimeError(
            "未能从 run_benchmark.py 输出中找到运行目录。"
            "请检查其“预测完成：...”输出。"
        )

    predictions = run_dir / "predictions.jsonl"
    if not predictions.is_file():
        raise FileNotFoundError(f"预测文件不存在：{predictions}")

    run_command([
        sys.executable, "evaluate.py",
        "--predictions", str(predictions),
        "--task", task,
    ])

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"评估结果不存在：{metrics_path}")

    with metrics_path.open(encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "model": model,
        "task": task,
        "run_dir": str(run_dir),
        "predictions": str(predictions),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def numeric_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        if key in COUNT_KEYS:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            result[key] = value
    return result


def build_comparisons(results):
    comparisons = []
    for task, pair in results.items():
        base = numeric_metrics(pair["base"]["metrics"])
        ft = numeric_metrics(pair["finetuned"]["metrics"])

        common_keys = sorted(set(base) & set(ft))
        metric_rows = []

        for key in common_keys:
            b = base[key]
            f = ft[key]
            metric_rows.append({
                "metric": key,
                "base_value": b,
                "finetuned_value": f,
                "delta": f - b,
            })

        comparisons.append({
            "task": task,
            "base_model": pair["base"]["model"],
            "finetuned_model": pair["finetuned"]["model"],
            "base_run_dir": pair["base"]["run_dir"],
            "finetuned_run_dir": pair["finetuned"]["run_dir"],
            "metrics": metric_rows,
        })
    return comparisons


def write_reports(report_dir, metadata, results, comparisons):
    report_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "metadata": metadata,
        "results": results,
        "comparisons": comparisons,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (report_dir / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fields = [
            "task", "base_model", "finetuned_model", "metric",
            "base_value", "finetuned_value", "delta",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            for metric in comparison["metrics"]:
                writer.writerow({
                    "task": comparison["task"],
                    "base_model": comparison["base_model"],
                    "finetuned_model": comparison["finetuned_model"],
                    **metric,
                })

    lines = [
        "# 基座模型与微调模型对比报告",
        "",
        f"- 运行时间：{metadata['created_at']}",
        f"- 评估范围：{metadata['scope']}",
        f"- 基座模型：{metadata['base_model']}",
        f"- 微调模型：{metadata['finetuned_model']}",
        f"- 样本限制：{metadata['limit']}",
        "",
        "> delta = 微调模型指标 - 基座模型指标。"
        "正负方向是否代表改善取决于具体指标；"
        "不同领域的指标不可直接混合求平均。",
        "",
    ]

    for comparison in comparisons:
        lines += [
            f"## {comparison['task']}",
            "",
            "| 指标 | 基座模型 | 微调模型 | 差值 |",
            "|---|---:|---:|---:|",
        ]
        if not comparison["metrics"]:
            lines.append("| 没有共同的数值指标 | — | — | — |")
        for metric in comparison["metrics"]:
            lines.append(
                f"| {metric['metric']} | {metric['base_value']} "
                f"| {metric['finetuned_value']} | {metric['delta']:+.6g} |"
            )
        lines += [
            "",
            f"- 基座结果：`{comparison['base_run_dir']}`",
            f"- 微调结果：`{comparison['finetuned_run_dir']}`",
            "",
        ]

    (report_dir / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )



def print_console_report(metadata, comparisons, report_dir):
    """将模型比较结果以表格形式打印到终端。"""
    print("\n" + "=" * 76)
    print("                 基座模型 vs 微调模型评估报告")
    print("=" * 76)
    print(f"基座模型：{metadata['base_model']}")
    print(f"微调模型：{metadata['finetuned_model']}")
    print(f"评估范围：{metadata['scope']}")
    print(f"样本限制：{metadata['limit'] or '完整数据集'}")

    for item in comparisons:
        print("\n" + "-" * 76)
        print(f"评估领域：{item['task']}")
        print("-" * 76)
        print(f"{'指标':<24}{'基座模型':>15}{'微调模型':>15}{'变化':>15}")
        print("-" * 76)

        if not item["metrics"]:
            print("没有可比较的共同数值指标。")
            continue

        for metric in item["metrics"]:
            name = metric["metric"]
            base = metric["base_value"]
            ft = metric["finetuned_value"]
            delta = metric["delta"]

            print(
                f"{name:<24}"
                f"{base:>15.4f}"
                f"{ft:>15.4f}"
                f"{delta:>+15.4f}"
            )

        # BLEU、ROUGE、准确率等一般是越高越好。
        # 不对未知指标擅自判断方向。
        higher_is_better = {
            "bleu", "rouge1", "rouge2", "rougeL",
            "rouge_1", "rouge_2", "rouge_l",
            "accuracy", "single_choice_accuracy",
            "multiple_choice_accuracy",
            "strict_acc", "flexible_acc",
        }
        recognized = [
            m for m in item["metrics"]
            if m["metric"].lower() in higher_is_better
        ]
        if recognized:
            improved = sum(m["delta"] > 0 for m in recognized)
            declined = sum(m["delta"] < 0 for m in recognized)
            unchanged = sum(m["delta"] == 0 for m in recognized)
            print(
                f"\n已识别的越高越好指标："
                f"{improved} 项提高，{declined} 项下降，"
                f"{unchanged} 项不变。"
            )

    print("\n" + "=" * 76)
    print("报告文件")
    print(f"目录：{report_dir}")
    print(f"CSV：{report_dir / 'summary.csv'}")
    print(f"JSON：{report_dir / 'summary.json'}")
    print(f"Markdown：{report_dir / 'summary.md'}")
    print("=" * 76)

def main():
    parser = argparse.ArgumentParser(
        description="一键比较基座模型和微调模型的 Benchmark 结果"
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--finetuned-model", required=True)
    parser.add_argument(
        "--scope", choices=["domain", "all"], default="domain"
    )
    parser.add_argument("--task", help="指定目标任务；domain 模式可用")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只显示计划，不加载模型或运行推理"
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于等于 1")

    try:
        task_cfg, models = load_configuration()
        tasks = resolve_tasks(args, task_cfg, models)

        print("\n评估计划")
        print("基座模型：", args.base_model)
        print("微调模型：", args.finetuned_model)
        print("任务：", ", ".join(tasks))
        print("每个模型每个任务的样本限制：", args.limit or "完整数据集")

        if args.dry_run:
            print("Dry run：未加载模型，未运行推理。")
            return

        results = {}
        for task in tasks:
            print(f"\n{'=' * 18} {task} {'=' * 18}")
            results[task] = {
                "base": run_one(args.base_model, task, args.limit),
                "finetuned": run_one(args.finetuned_model, task, args.limit),
            }

        comparisons = build_comparisons(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = REPORT_ROOT / f"{timestamp}_{args.scope}"

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scope": args.scope,
            "base_model": args.base_model,
            "finetuned_model": args.finetuned_model,
            "tasks": tasks,
            "limit": args.limit,
        }

        write_reports(report_dir, metadata, results, comparisons)
        print_console_report(metadata, comparisons, report_dir)

        print("\n评估完成。")
        print("汇总目录：", report_dir)
        print("CSV：", report_dir / "summary.csv")
        print("JSON：", report_dir / "summary.json")
        print("Markdown：", report_dir / "summary.md")

    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
