"""math_reasoning 评估器边界测试（说明书第 10 步）。

运行：python scripts/test_math_evaluator.py
每条用例手工计算预期结果，断言与 evaluate() 输出一致；
任何改动 evaluators/math.py 的提交都应先通过本脚本。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluators.math import evaluate  # noqa: E402


def make_predictions(rows):
    return [
        {"sample_id": f"s{i}", "status": status, "prediction": pred,
         "reference": ref, "metadata": {}}
        for i, (status, pred, ref) in enumerate(rows)
    ]


def expect(name, condition):
    if not condition:
        raise SystemExit(f"失败用例：{name}")
    print(f"[OK] {name}")


def run_case(name, rows, check):
    metrics = evaluate(make_predictions(rows))
    print(f"  -> {name}: strict_acc={metrics['strict_acc']} "
          f"flexible_acc={metrics['flexible_acc']}")
    check(metrics)


def main():
    # 1 正确：SFT 口径“答案是 X。”，数值等于 reference。
    run_case(
        "轻微正确答案",
        [("success", "逐步解答: 16-3-4=9。9*2=18。答案是 18。", "18")],
        lambda m: expect(
            "sft 正确", m["strict_acc"] == 1.0 and m["flexible_acc"] == 1.0
        ),
    )
    # 2 错误答案。
    run_case(
        "错误答案",
        [("success", "逐步解答: 答案是 17。", "18")],
        lambda m: expect("错误计 0 分", m["strict_acc"] == 0.0),
    )
    # 3 空输出：strict 解析失败按错误计，分母仍是有效样本。
    run_case(
        "空输出",
        [("success", "", "18")],
        lambda m: expect(
            "空输出计 0 分", m["strict_acc"] == 0.0 and m["num_scored"] == 1
        ),
    )
    # 4 不可解析纯文本：无答案行、无数字。
    run_case(
        "不可解析",
        [("success", "这个题目我不会做，无法给出结论。", "18")],
        lambda m: expect("无数字计 0 分", m["strict_acc"] == 0.0),
    )
    # 5 复读下一题：在“问题:”处截断，复读答案是 99 不影响抽取 42。
    run_case(
        "复读截断",
        [("success", "逐步解答: 答案是 42。\n\n问题: 某某问题\n逐步解答:答案是 99。", "42")],
        lambda m: expect(
            "复读不计分", m["strict_acc"] == 1.0 and m["flexible_acc"] == 1.0
        ),
    )
    # 6 数值比较：2.5 与 2.50、40.0 与 40 视为相同。
    run_case(
        "数值等价",
        [("success", "答案是 2.50。", "2.5"),
         ("success", "答案是 40。", "40.0")],
        lambda m: expect("float 等价命中", m["strict_acc"] == 1.0),
    )
    # 7 旧格式兼容：demo 题的“最终答案：X”。
    run_case(
        "旧格式",
        [("success", "12*13=156。最终答案：156。", "156")],
        lambda m: expect("legacy 命中", m["strict_acc"] == 1.0),
    )
    # 8 无句号 / “答案为 X”：strict 不可解析，flexible 兜底命中。
    run_case(
        "flexible 兜底",
        [("success", "逐步解答: 因此结果为 18。", "18")],
        lambda m: expect(
            "flex 兜底", m["strict_acc"] == 0.0 and m["flexible_acc"] == 1.0
        ),
    )
    # 9 status=error 的样本进入 num_failed，不进入分母。
    run_case(
        "失败样本分母",
        [("error", None, "18"), ("success", "答案是 18。", "18")],
        lambda m: expect(
            "分母与失败数",
            m["num_samples"] == 2 and m["num_scored"] == 1
            and m["num_failed"] == 1 and m["strict_acc"] == 1.0,
        ),
    )
    # 10 无任何有效样本（status 全非 success）时为 None 而非除零。
    metrics = evaluate([{"sample_id": "s", "status": "error", "prediction": None, "reference": "1"}])
    expect(
        "空集 None", metrics["strict_acc"] is None and metrics["flexible_acc"] is None
        and metrics["num_scored"] == 0,
    )

    # 11 输出只保留 strict_acc / flexible_acc 两项指标，外加样本记录字段。
    metrics = evaluate(make_predictions([("success", "答案是 1。", "1")]))
    allowed = {
        "num_samples", "num_scored", "num_failed",
        "strict_acc", "flexible_acc", "metric_note",
    }
    expect("仅两项指标", not set(metrics) - allowed)

    print("全部用例通过。")


if __name__ == "__main__":
    main()
