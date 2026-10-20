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
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no}: {e}")
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="仅运行前 N 条，用于冒烟测试")
    args = parser.parse_args()

    models_cfg = load_yaml(ROOT / "configs/models.yaml")["models"]
    bench_cfg = load_yaml(ROOT / "configs/benchmark.yaml")
    if args.model not in models_cfg:
        raise SystemExit(f"未知模型 {args.model}；请检查 configs/models.yaml")
    if args.task not in bench_cfg["tasks"]:
        raise SystemExit(f"未知任务 {args.task}；请检查 configs/benchmark.yaml")

    model_cfg = models_cfg[args.model]
    task_cfg = bench_cfg["tasks"][args.task]
    if args.task not in model_cfg.get("tasks", []):
        raise SystemExit(f"模型 {args.model} 未登记参加任务 {args.task}")

    dataset_path = ROOT / task_cfg["dataset"]
    prompt_path = ROOT / task_cfg["prompt_template"]
    data = read_jsonl(dataset_path)
    if args.limit is not None:
        data = data[:args.limit]
    template = prompt_path.read_text(encoding="utf-8")

    backend = model_cfg["backend"].lower()
    if backend == "transformers":
        adapter = TransformersAdapter(
            base_model=model_cfg["base_model"],
            adapter_path=model_cfg.get("adapter_path"),
            device_map=model_cfg.get("device_map", "auto"),
            trust_remote_code=model_cfg.get("trust_remote_code", False),
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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = ROOT / "outputs" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    run_config = {
        "run_id": run_id,
        "model_id": args.model,
        "task": args.task,
        "backend": backend,
        "dataset_version": bench_cfg.get("dataset_version"),
        "dataset_path": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": sha256_file(dataset_path),
        "prompt_path": str(prompt_path.relative_to(ROOT)),
        "prompt_sha256": sha256_file(prompt_path),
        "generation_config": bench_cfg.get("generation", {}),
        "model_config": model_cfg,
        "num_samples_requested": len(data),
        "python_version": platform.python_version(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    failed = 0
    try:
        adapter.load()
        with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for idx, item in enumerate(data, 1):
                record = {
                    "sample_id": item["id"],
                    "model_id": args.model,
                    "task": args.task,
                }
                try:
                    prompt = template.replace("{input}", item["input"])
                    start = time.perf_counter()
                    prediction = adapter.generate(
                        prompt, bench_cfg.get("generation", {})
                    )
                    elapsed = (time.perf_counter() - start) * 1000
                    record.update({
                        "prediction": prediction,
                        "status": "success",
                        "latency_ms": round(elapsed, 2),
                    })
                except Exception as exc:
                    failed += 1
                    record.update({
                        "prediction": None,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{idx}/{len(data)}] {item['id']}: {record['status']}")
    finally:
        adapter.close()

    run_config["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_config["num_failed"] = failed
    run_config["num_success"] = len(data) - failed
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"预测完成：{out_dir}")
    print(f"成功：{len(data)-failed} / {len(data)}；失败：{failed}")
    print("下一步运行 evaluate.py 对预测结果评分。")


if __name__ == "__main__":
    main()
