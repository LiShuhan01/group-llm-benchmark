"""把 MGSM zh test 集转换为 group-llm-benchmark 统一 JSONL 格式。

数据源（按优先级）：
1. --source 显式指定的 .parquet 或 .jsonl 文件（推荐：离线、可追溯）；
2. 未指定时依次自动探测：
   a) 项目内 evalweave 已准备的同源数据 ../evalweave/data/mgsm_zh/test_v1.jsonl
   b) HF 镜像下载官方 parquet（联网通道，较慢，结果与离线缓存同序同源）

原始数据：huggingface.co/datasets/juletxara/mgsm 配置 zh、split test，共 250 题，
字段 question / answer_number。本脚本使用全量 test 集，不做抽样、不设随机种子。

输出：data/math_reasoning/mgsm_zh_test_v1.jsonl
    {"id":"mgsm_zh_test_0000","task":"math_reasoning",
     "input":"<题面>","reference":"<答案数值>",
     "metadata":{"source":"mgsm_zh","split":"test",
                 "dataset_version":"mgsm-zh-test-v1"}}

同一来源与行序生成的文件 id 与 benchmarks/evalweave 的 mgsm 数据完全一致，
两套评测流程可逐题对照。id 格式固定，修改字段映射会破坏可比性，请勿改动。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "math_reasoning" / "mgsm_zh_test_v1.jsonl"
EXPECTED_TASK = "math_reasoning"
MIRROR_URL = (
    "https://hf-mirror.com/datasets/juletxara/mgsm/resolve/main/zh/"
    "test-00000-of-00001.parquet"
)
EVALWEAVE_SOURCE = ROOT.parent / "evalweave" / "data" / "mgsm_zh" / "test_v1.jsonl"


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_from_parquet(path):
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pylist()


def rows_from_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                rows.append(json.loads(line))
                # evalweave 源文件的行序即官方 test 行序，此处仅透传。
    return rows


def load_rows(source):
    source = Path(source)
    if not source.is_file():
        raise SystemExit(f"数据源不存在：{source}")
    if source.suffix.lower() == ".parquet":
        return rows_from_parquet(source)
    return rows_from_jsonl(source)


def auto_source():
    """未显式指定时：先找 evalweave 同源数据，最后才联网下载。"""
    if EVALWEAVE_SOURCE.is_file():
        print(f"[prep] 使用 evalweave 同源数据：{EVALWEAVE_SOURCE}")
        return EVALWEAVE_SOURCE
    print("[prep] 未找到本地源，从 HF 镜像下载官方 parquet……")
    import requests

    cache = ROOT / "data" / "math_reasoning" / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "mgsm_zh_test.parquet"
    if not target.is_file():
        resp = requests.get(MIRROR_URL, timeout=120)
        resp.raise_for_status()
        target.write_bytes(resp.content)
    print(f"[prep] 已缓存：{target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        default=None,
        help="MGSM zh test 的 .parquet 或 .jsonl；缺省时自动探测",
    )
    args = parser.parse_args()

    source = args.source if args.source else auto_source()
    rows = load_rows(source)
    print(f"[prep] 源数据行数：{len(rows)}")

    records = []
    seen = set()
    for index, row in enumerate(rows):
        question = str(row.get("question", "")).strip()
        # test 集 answer 列为空，答案数值统一取 answer_number。
        answer = row.get("answer_number")
        reference = str(answer).strip() if answer is not None else ""
        if not question:
            raise SystemExit(f"第 {index} 行 question 为空")
        if not reference:
            raise SystemExit(f"第 {index} 行 answer_number 为空")
        try:
            float(reference.replace(",", ""))
        except ValueError:
            raise SystemExit(
                f"第 {index} 行 answer_number 不是数值：{reference!r}"
            )
        sample_id = f"mgsm_zh_test_{index:04d}"
        if sample_id in seen:
            raise SystemExit(f"ID 重复：{sample_id}")
        seen.add(sample_id)
        records.append(
            {
                "id": sample_id,
                "task": EXPECTED_TASK,
                "input": question,
                "reference": reference.replace(",", ""),
                "metadata": {
                    "source": "mgsm_zh",
                    "split": "test",
                    "dataset_version": "mgsm-zh-test-v1",
                },
            }
        )

    if len(records) != 250:
        print(
            f"[prep] 警告：官方 test 集应为 250 题，实际 {len(records)} 条，请人工确认"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"[prep] 已写出 {len(records)} 条 -> {OUT_PATH}")
    print(f"[prep] sha256={sha256_file(OUT_PATH)}")


if __name__ == "__main__":
    sys.exit(main())
