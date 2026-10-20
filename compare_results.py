import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent

NON_METRICS = {
    "num_samples",
    "num_scored",
    "num_failed",
    "num_success",
    "num_samples_requested",
    "num_not_run",
}


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_metric_value(key, value):
    return (
        key not in NON_METRICS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def load_runs(runs_dir):
    runs = []

    for config_path in sorted(runs_dir.glob("*/run_config.json")):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.json"

        if not metrics_path.is_file():
            continue

        try:
            config = read_json(config_path)
            metrics = read_json(metrics_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[跳过] {run_dir}: 无法读取记录：{exc}")
            continue

        model_id = config.get("model_id")
        task = config.get("task") or metrics.get("task")

        if not model_id or not task:
            print(f"[跳过] {run_dir}: 缺少 model_id 或 task")
            continue

        values = {
            key: value
            for key, value in metrics.items()
            if is_metric_value(key, value)
        }

        if not values:
            print(f"[提示] {run_dir}: 没有可汇总的数值指标")
            continue

        runs.append({
            "run_id": config.get("run_id", run_dir.name),
            "run_dir": str(run_dir),
            "model_id": model_id,
            "model_role": config.get("model_role", "unspecified"),
            "base_model_id": config.get("base_model_id"),
            "task": task,
            "backend": config.get("backend"),
            "dataset_version": config.get("dataset_version"),
            "dataset_sha256": config.get("dataset_sha256"),
            "status": config.get("status", "unknown"),
            "num_success": config.get("num_success"),
            "num_failed": config.get("num_failed"),
            "metrics": values,
        })

    return runs


def aggregate_runs(runs):
    """按模型、任务、数据集内容和指标汇总重复运行。"""
    groups = {}

    for run in runs:
        for metric, score in run["metrics"].items():
            key = (
                run["model_id"],
                run["model_role"],
                run["base_model_id"],
                run["task"],
                run["dataset_sha256"],
                metric,
            )
            groups.setdefault(key, []).append((run, score))

    summaries = []

    for key, entries in sorted(groups.items(), key=lambda item: str(item[0])):
        model_id, role, base_id, task, dataset_hash, metric = key
        scores = [score for _, score in entries]
        first = entries[0][0]

        summaries.append({
            "model_id": model_id,
            "model_role": role,
            "base_model_id": base_id,
            "task": task,
            "metric": metric,
            "dataset_version": first["dataset_version"],
            "dataset_sha256": dataset_hash,
            "num_runs": len(scores),
            "mean": statistics.mean(scores),
            "stddev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        })

    return summaries


def build_rows(runs):
    """保留逐次运行明细，并为微调模型计算基座均值对照。"""
    base_groups = {}

    for summary in aggregate_runs(runs):
        if summary["model_role"] != "base":
            continue

        key = (
            summary["model_id"],
            summary["task"],
            summary["dataset_sha256"],
            summary["metric"],
        )
        base_groups[key] = summary

    rows = []

    for run in runs:
        for metric, score in run["metrics"].items():
            row = {
                "run_id": run["run_id"],
                "model_id": run["model_id"],
                "model_role": run["model_role"],
                "base_model_id": run["base_model_id"],
                "task": run["task"],
                "metric": metric,
                "score": score,
                "base_score": None,
                "delta": None,
                "relative_change_pct": None,
                "dataset_version": run["dataset_version"],
                "dataset_sha256": run["dataset_sha256"],
                "backend": run["backend"],
                "status": run["status"],
                "num_success": run["num_success"],
                "num_failed": run["num_failed"],
                "run_dir": run["run_dir"],
            }

            if run["model_role"] == "finetuned" and run["base_model_id"]:
                key = (
                    run["base_model_id"],
                    run["task"],
                    run["dataset_sha256"],
                    metric,
                )
                base = base_groups.get(key)

                if base is not None:
                    base_score = base["mean"]
                    row["base_score"] = base_score
                    row["delta"] = score - base_score

                    if base_score != 0:
                        row["relative_change_pct"] = (
                            row["delta"] / abs(base_score) * 100
                        )

            rows.append(row)

    return rows


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, runs, rows, summaries):
    lines = [
        "# Group LLM Benchmark 结果汇总",
        "",
        f"- 有效实验数：{len(runs)}",
        f"- 逐次指标记录数：{len(rows)}",
        f"- 模型汇总记录数：{len(summaries)}",
        "",
        "## 模型多次运行汇总",
        "",
        "| 任务 | 模型 | 角色 | 指标 | 运行次数 | 均值 | 标准差 | 最小值 | 最大值 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for item in summaries:
        lines.append(
            f"| {item['task']} | {item['model_id']} | "
            f"{item['model_role']} | {item['metric']} | "
            f"{item['num_runs']} | {item['mean']:.6f} | "
            f"{item['stddev']:.6f} | {item['min']:.6f} | "
            f"{item['max']:.6f} |"
        )

    lines.extend([
        "",
        "## 基座与微调模型逐次对比",
        "",
        "| 任务 | 模型 | 指标 | 本次得分 | 基座均值 | 差值 | 相对变化 |",
        "|---|---|---|---:|---:|---:|---:|",
    ])

    for row in rows:
        base = f"{row['base_score']:.6f}" if row["base_score"] is not None else "—"
        delta = f"{row['delta']:+.6f}" if row["delta"] is not None else "—"
        pct = (
            f"{row['relative_change_pct']:+.2f}%"
            if row["relative_change_pct"] is not None else "—"
        )
        lines.append(
            f"| {row['task']} | {row['model_id']} | {row['metric']} | "
            f"{row['score']:.6f} | {base} | {delta} | {pct} |"
        )

    lines.extend([
        "",
        "## 解读注意事项",
        "",
        "- 差值 = 微调模型本次得分 - 对应基座模型的多次运行均值。",
        "- 只有任务、数据集内容哈希和指标名称一致时才配对。",
        "- 标准差描述重复运行得分的离散程度；只有一次运行时标准差记为 0。",
        "- 正向变化是否代表更好取决于指标方向，本工具不自行判断。",
        "- 不同任务、不同指标之间不可直接相加或排名。",
        "- 演示数据只能用于流程验证，不能作为正式能力结论。",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="通用 Benchmark 结果汇总与基座/微调模型比较"
    )
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--output-dir", default="outputs/comparison")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = ROOT / runs_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(runs_dir)
    rows = build_rows(runs)
    summaries = aggregate_runs(runs)

    if not rows:
        raise SystemExit(f"没有找到可汇总的实验指标：{runs_dir}")

    detail_columns = [
        "run_id", "model_id", "model_role", "base_model_id", "task",
        "metric", "score", "base_score", "delta", "relative_change_pct",
        "dataset_version", "dataset_sha256", "backend", "status",
        "num_success", "num_failed", "run_dir",
    ]
    summary_columns = [
        "model_id", "model_role", "base_model_id", "task", "metric",
        "dataset_version", "dataset_sha256", "num_runs", "mean",
        "stddev", "min", "max",
    ]

    csv_path = output_dir / "comparison.csv"
    summary_path = output_dir / "model_summary.csv"
    report_path = output_dir / "report.md"

    write_csv(csv_path, rows, detail_columns)
    write_csv(summary_path, summaries, summary_columns)
    write_report(report_path, runs, rows, summaries)

    paired = sum(row["delta"] is not None for row in rows)

    print(f"发现实验：{len(runs)}")
    print(f"逐次指标记录：{len(rows)}")
    print(f"模型汇总记录：{len(summaries)}")
    print(f"已配对的基座/微调指标：{paired}")
    print(f"逐次明细：{csv_path}")
    print(f"模型汇总：{summary_path}")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
