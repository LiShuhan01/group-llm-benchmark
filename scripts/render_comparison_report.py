#!/usr/bin/env python3
"""读取比较 JSON，打印清晰的终端汇总并生成 Markdown 报告。"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = ROOT / "outputs" / "comparisons"


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_report(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    for comp in obj.get("comparisons", []):
        task = comp.get("task", "unknown")
        for metric in comp.get("metrics", []):
            rows.append({
                "task": task,
                "metric": metric.get("metric", "unknown"),
                "base": metric.get("base_value"),
                "finetuned": metric.get("finetuned_value"),
                "delta": metric.get("delta"),
            })

    # 兼容比较 JSON 中有 results、但没有 comparisons 明细的情况
    if not rows:
        for task, result in obj.get("results", {}).items():
            base = result.get("base", {}).get("metrics", {})
            ft = result.get("finetuned", {}).get("metrics", {})
            for metric in sorted(set(base) & set(ft)):
                if isinstance(base[metric], (int, float)) and isinstance(
                    ft[metric], (int, float)
                ):
                    rows.append({
                        "task": task,
                        "metric": metric,
                        "base": base[metric],
                        "finetuned": ft[metric],
                        "delta": ft[metric] - base[metric],
                    })

    return obj, rows


def main():
    parser = argparse.ArgumentParser(
        description="将 Benchmark 比较 JSON 转为终端表格和 Markdown 报告"
    )
    parser.add_argument(
        "--file", type=Path,
        help="指定比较 JSON；默认自动选择最新的比较 JSON"
    )
    args = parser.parse_args()

    if args.file:
        path = args.file.expanduser().resolve()
    else:
        candidates = list(COMPARISONS.glob("**/*.json"))
        if not candidates:
            raise SystemExit(
                f"没有找到比较 JSON：{COMPARISONS}\n"
                "请等待比较任务产生结果，或使用 --file 指定文件。"
            )
        path = max(candidates, key=lambda p: p.stat().st_mtime)

    if not path.is_file():
        raise SystemExit(f"文件不存在：{path}")

    obj, rows = load_report(path)
    metadata = obj.get("metadata", {})
    base_name = metadata.get("base_model", "base")
    ft_name = metadata.get("finetuned_model", "finetuned")

    print(f"\n比较报告：{path}")
    print(f"基座模型：{base_name}")
    print(f"微调模型：{ft_name}")
    print(f"评估范围：{metadata.get('scope', 'unknown')}")
    print()

    headers = ["领域", "指标", "基座模型", "微调模型", "差值"]
    data = [
        [r["task"], r["metric"], fmt(r["base"]),
         fmt(r["finetuned"]), fmt(r["delta"])]
        for r in rows
    ]

    if data:
        widths = [
            max(len(headers[i]), *(len(str(row[i])) for row in data))
            for i in range(len(headers))
        ]
        line = "-+-".join("-" * w for w in widths)
        print(" | ".join(headers[i].ljust(widths[i]) for i in range(5)))
        print(line)
        for row in data:
            print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(5)))
    else:
        print("该 JSON 中尚未找到可展示的指标。")

    results = obj.get("results", {})
    failed = []
    for task, result in results.items():
        for role in ("base", "finetuned"):
            item = result.get(role, {})
            metrics = item.get("metrics", {})
            if not item or not metrics:
                failed.append(f"{task}/{role}: 尚无完整指标")
            elif metrics.get("num_failed", 0):
                failed.append(
                    f"{task}/{role}: 失败 {metrics['num_failed']} 条"
                )

    if failed:
        print("\n需要关注：")
        for item in failed:
            print(f"  - {item}")

    md = [
        "# Benchmark 模型对比报告", "",
        f"- 基座模型：`{base_name}`",
        f"- 微调模型：`{ft_name}`",
        f"- 评估范围：`{metadata.get('scope', 'unknown')}`",
        f"- 样本限制：`{metadata.get('limit', '未记录')}`",
        f"- 原始 JSON：`{path}`", "",
        "| 领域 | 指标 | 基座模型 | 微调模型 | 差值（微调 - 基座） |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['task']} | {r['metric']} | {fmt(r['base'])} "
            f"| {fmt(r['finetuned'])} | {fmt(r['delta'])} |"
        )

    md += ["", "## 运行情况", ""]
    if results:
        md.append("| 领域 | 模型角色 | 成功评分样本 | 失败样本 | 指标文件 |")
        md.append("|---|---|---:|---:|---|")
        for task, result in results.items():
            for role in ("base", "finetuned"):
                item = result.get(role, {})
                metrics = item.get("metrics", {})
                if not item:
                    md.append(f"| {task} | {role} | N/A | N/A | 未生成 |")
                    continue
                md.append(
                    f"| {task} | {role} | "
                    f"{metrics.get('num_scored', 'N/A')} | "
                    f"{metrics.get('num_failed', 'N/A')} | "
                    f"`{item.get('metrics_path', 'N/A')}` |"
                )
    else:
        md.append("该比较文件未包含逐任务运行详情。")

    md += [
        "", "> 注意：不同任务的指标定义和量纲可能不同，"
        "差值只适用于同一任务、同一指标的模型比较。", ""
    ]

    output = path.parent / "report.md"
    output.write_text("\n".join(md), encoding="utf-8")
    print(f"\nMarkdown 报告已保存：{output}")


if __name__ == "__main__":
    main()
