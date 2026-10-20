import argparse
import hashlib
import json
import platform
import time
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from adapters.transformers_adapter import TransformersAdapter
from adapters.vllm_adapter import VLLMAdapter

ROOT = Path(__file__).resolve().parent


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc

            if "id" not in item or "input" not in item:
                raise ValueError(
                    f"{path}:{line_no}: 每条样本必须包含 id 和 input"
                )
            rows.append(item)
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def redact_config(config):
    """避免把 API 密钥等敏感字段写入实验元数据。"""
    if isinstance(config, dict):
        result = {}
        for key, value in config.items():
            if key.lower() in {
                "api_key", "token", "password", "secret"
            }:
                result[key] = "***REDACTED***"
            else:
                result[key] = redact_config(value)
        return result
    if isinstance(config, list):
        return [redact_config(item) for item in config]
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅运行前 N 条，用于冒烟测试",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit 必须大于 0")

    models_cfg = load_yaml(ROOT / "configs/models.yaml")["models"]
    bench_cfg = load_yaml(ROOT / "configs/benchmark.yaml")

    if args.model not in models_cfg:
        raise SystemExit(
            f"未知模型 {args.model}；请检查 configs/models.yaml"
        )

    tasks_cfg = bench_cfg.get("tasks", {})
    if args.task not in tasks_cfg:
        raise SystemExit(
            f"未知任务 {args.task}；请检查 configs/benchmark.yaml"
        )

    model_cfg = models_cfg[args.model]
    task_cfg = tasks_cfg[args.task]

    if not task_cfg.get("enabled", True):
        raise SystemExit(
            f"任务 {args.task} 当前未启用，请先检查 benchmark.yaml"
        )

    if args.task not in model_cfg.get("tasks", []):
        raise SystemExit(
            f"模型 {args.model} 未登记参加任务 {args.task}"
        )

    # 兼容 dataset_path 与旧字段 dataset。
    dataset_value = task_cfg.get("dataset_path", task_cfg.get("dataset"))
    if not dataset_value:
        raise SystemExit(f"任务 {args.task} 没有配置测试集路径")

    dataset_path = resolve_path(dataset_value)

    # 优先使用任务显式指定的 Prompt；否则自动按任务名寻找。
    prompt_value = task_cfg.get(
        "prompt_template",
        f"prompts/{args.task}.txt",
    )
    prompt_path = resolve_path(prompt_value)

    for path in (dataset_path, prompt_path):
        if not path.is_file():
            raise SystemExit(f"文件不存在：{path}")

    data = read_jsonl(dataset_path)

    if args.limit is not None:
        data = data[:args.limit]

    if not data:
        raise SystemExit("测试集没有可运行的样本")

    template = prompt_path.read_text(encoding="utf-8")

    # 兼容新版 inference 和旧版 generation。
    inference_cfg = dict(
        bench_cfg.get("inference", bench_cfg.get("generation", {})) or {}
    )

    # seed 是实验设置，不是模型 generate() 的参数。
    generation_cfg = {
        key: inference_cfg[key]
        for key in (
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "num_beams",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "length_penalty",
        )
        if key in inference_cfg
    }
    generation_cfg.setdefault("max_new_tokens", 256)
    generation_cfg.setdefault("do_sample", False)

    backend = model_cfg["backend"].lower()

    if backend == "transformers":
        adapter = TransformersAdapter(
            base_model=model_cfg["base_model"],
            adapter_path=model_cfg.get("adapter_path"),
            device_map=model_cfg.get("device_map", "auto"),
            trust_remote_code=model_cfg.get(
                "trust_remote_code", False
            ),
        )
    elif backend == "vllm":
        adapter = VLLMAdapter(
            api_base=model_cfg["api_base"],
            served_model_name=model_cfg["served_model_name"],
            api_key=model_cfg.get("api_key", "EMPTY"),
            timeout=model_cfg.get("timeout", 300),
        )
    else:
        raise SystemExit(f"不支持的 backend: {backend}")

    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:6]
    )

    output_cfg = bench_cfg.get("output", {})
    output_root = resolve_path(
        output_cfg.get("root_dir", "outputs/runs")
    )
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    run_config = {
        "benchmark_name": bench_cfg.get("benchmark", {}).get(
            "name", "group-llm-benchmark"
        ),
        "benchmark_version": bench_cfg.get("benchmark", {}).get(
            "version", "unknown"
        ),
        "run_id": run_id,
        "model_id": args.model,
        "model_role": model_cfg.get("role", "unspecified"),
        "base_model_id": model_cfg.get("base_model_id"),
        "base_model_path": model_cfg.get("base_model"),
        "adapter_path": model_cfg.get("adapter_path"),
        "task": args.task,
        "backend": backend,
        "dataset_version": task_cfg.get(
            "dataset_version",
            bench_cfg.get("dataset_version", "unspecified"),
        ),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "inference_config": inference_cfg,
        "generation_config": generation_cfg,
        "model_config": redact_config(model_cfg),
        "num_samples_requested": len(data),
        "python_version": platform.python_version(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
    }

    metadata_path = out_dir / "run_config.json"
    save_json(metadata_path, run_config)

    succeeded = 0
    failed = 0
    fatal_error = None
    start_run = time.perf_counter()

    try:
        adapter.load()

        with (out_dir / "predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for idx, item in enumerate(data, 1):
                record = {
                    "sample_id": item["id"],
                    "model_id": args.model,
                    "task": args.task,
                }

                try:
                    prompt = template.replace(
                        "{input}", str(item["input"])
                    )
                    start = time.perf_counter()
                    prediction = adapter.generate(
                        prompt, generation_cfg
                    )
                    elapsed = (time.perf_counter() - start) * 1000

                    record.update({
                        "prediction": prediction,
                        "status": "success",
                        "latency_ms": round(elapsed, 2),
                    })
                    succeeded += 1
                except Exception as exc:
                    failed += 1
                    record.update({
                        "prediction": None,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[{idx}/{len(data)}] "
                    f"{item['id']}: {record['status']}"
                )

    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        raise

    finally:
        try:
            adapter.close()
        finally:
            run_config.update({
                "finished_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "duration_seconds": round(
                    time.perf_counter() - start_run, 3
                ),
                "num_failed": failed,
                "num_success": succeeded,
                "num_not_run": max(
                    0, len(data) - succeeded - failed
                ),
                "status": (
                    "failed" if fatal_error
                    else "completed_with_errors" if failed
                    else "completed"
                ),
            })
            if fatal_error:
                run_config["fatal_error"] = fatal_error
            save_json(metadata_path, run_config)

    print(f"预测完成：{out_dir}")
    not_run = max(0, len(data) - succeeded - failed)
    print(
        f"计划：{len(data)}；成功：{succeeded}；"
        f"失败：{failed}；未执行：{not_run}"
    )
    print(f"实验记录：{metadata_path}")
    print("下一步运行 evaluate.py 对预测结果评分。")


if __name__ == "__main__":
    main()
