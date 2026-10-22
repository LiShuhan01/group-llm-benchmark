#!/usr/bin/env python3
import importlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    config_path = ROOT / "configs" / "benchmark.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    errors = []
    warnings = []
    checked = 0

    for task, cfg in config.get("tasks", {}).items():
        cfg = cfg or {}

        if not cfg.get("enabled", True):
            print(f"[跳过] {task}: 未启用")
            continue

        checked += 1
        print(f"\n[检查] {task}")

        dataset_value = cfg.get("dataset_path") or cfg.get("dataset")
        prompt_value = cfg.get(
            "prompt_template", f"prompts/{task}.txt"
        )
        evaluator_name = cfg.get("evaluator")

        def resolve(value):
            p = Path(value).expanduser()
            return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

        if not dataset_value:
            errors.append(f"{task}: 缺少 dataset_path")
            continue

        dataset_path = resolve(dataset_value)
        prompt_path = resolve(prompt_value)

        if not dataset_path.is_file():
            errors.append(f"{task}: 数据集不存在 {dataset_path}")
            continue

        if not prompt_path.is_file():
            errors.append(f"{task}: Prompt 不存在 {prompt_path}")

        if not evaluator_name:
            errors.append(f"{task}: 缺少 evaluator 配置")
        else:
            try:
                module = importlib.import_module(
                    f"evaluators.{evaluator_name}"
                )
                if not callable(getattr(module, "evaluate", None)):
                    errors.append(
                        f"{task}: evaluators/{evaluator_name}.py "
                        "缺少 evaluate()"
                    )
            except Exception as exc:
                errors.append(f"{task}: 评估器导入失败：{exc}")

        ids = set()
        count = 0

        try:
            with dataset_path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        errors.append(
                            f"{task}: JSON 第 {line_no} 行错误：{exc}"
                        )
                        continue

                    count += 1
                    for field in ("id", "input", "reference"):
                        if field not in item:
                            errors.append(
                                f"{task}: 第 {line_no} 行缺少 {field}"
                            )

                    sample_id = item.get("id")
                    if sample_id is not None:
                        if sample_id in ids:
                            errors.append(
                                f"{task}: 重复 ID {sample_id!r}"
                            )
                        ids.add(sample_id)

                    # 允许数据集原始任务标签与 Benchmark 任务 ID 不同。
                    # XMedBench 原始记录使用 medical_qa，
                    # Benchmark 注册任务名为 medical_mcq_xmedbench_v1。
                    task_aliases = {
                        "medical_mcq_xmedbench_v1": {"medical_qa"},
                    }
                    actual_task = item.get("task")
                    allowed_tasks = {task} | task_aliases.get(task, set())

                    if actual_task not in allowed_tasks:
                        warnings.append(
                            f"{task}: 第 {line_no} 行 task 字段为 "
                            f"{actual_task!r}"
                        )
        except OSError as exc:
            errors.append(f"{task}: 数据集读取失败：{exc}")

        if count == 0:
            errors.append(f"{task}: 数据集没有有效记录")
        print(f"  数据集样本数：{count}")
        print(f"  评估器：{evaluator_name}")
        print(f"  Prompt：{prompt_path}")

    print(f"\n检查任务数：{checked}")
    for warning in warnings:
        print(f"[警告] {warning}")
    for error in errors:
        print(f"[错误] {error}")

    if errors:
        print(f"\n检查未通过：{len(errors)} 个错误")
        return 1

    print("\n配置、数据集和评估器静态检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
