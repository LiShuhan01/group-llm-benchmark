"""math_reasoning 评估器：strict / flexible 双口径。

mgsm-zh-test-v1 数据集对应微调模型（Qwen2.5-3B + math3b QLoRA）的输出格式
“逐步解答: … 答案是 X。”，判分口径与 benchmarks/evalweave 已逐题对照验证的
eval_mgsm.py 保持一致：

1. 解析前先做两截断：生成文本在复述的下一个“问题:”处截断（复读不计入），
   再在首个完整答案行（“#### N” / “答案是 N。”）处截断；
2. strict：优先按“答案是 X。”正则抽取；旧格式回退链
   （“最终答案[:：]”、“答案[:：]”/“Answer[:：]”、\\boxed{...}）保持兼容；
3. flexible：截断后最后一个数字；
4. 与 reference（数值字符串）比较：双方均可解析为 float 时按数值比较
   （如 “42.0” 与 “42” 视为相同），否则按规范化字符串比较。

strict 抽不到时该样本计入 num_invalid_answer，但不计入失败样本
（flexible 仍可能命中，见 flexible_acc）。
"""

import re
from evaluators.common import successful

# 与 eval_mgsm.py 一致的三条正则。
FLEX_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
STRICT_RE = re.compile(r"答案是\s*(\-?[0-9\.\,]+)。")
ANS_LINE_RE = re.compile(
    r"####\s*-?\d[\d,]*(?:\.\d+)?|答案是\s*-?\d[\d,]*(?:\.\d+)?。"
)

# 旧格式回退链（demo 乘法题等：模型按旧 Prompt 写“最终答案：X”）。
LEGACY_PATTERNS = [
    re.compile(r"最终答案\s*[:：]\s*(.+)"),
    re.compile(r"(?:答案|Answer)\s*[:：]\s*(.+)"),
    re.compile(r"\\boxed\{([^{}]+)\}"),
]


def normalize_answer(text):
    text = (text or "").strip().lower()
    text = text.replace(",", "").replace("，", "")
    text = re.sub(r"\s+", "", text)
    return text.rstrip("。.!！")


def truncate_generation(text):
    """在复述的下一题与完整答案行处截断，复读内容不参与抽取。"""
    if not text:
        return ""
    text = str(text).split("问题:")[0]
    match = ANS_LINE_RE.search(text)
    if match:
        text = text[: match.end()]
    return text


def extract(text):
    """返回 (strict, flexible)；strict 可能为 None 表示不可解析。"""
    truncated = truncate_generation(text)
    match = STRICT_RE.search(truncated)
    if match:
        strict = normalize_answer(match.group(1))
    else:
        strict = None
        for pattern in LEGACY_PATTERNS:
            legacy = pattern.search(truncated)
            if legacy:
                strict = normalize_answer(
                    legacy.group(1).splitlines()[0]
                )
                break
    numbers = FLEX_RE.findall(truncated)
    flexible = normalize_answer(numbers[-1]) if numbers else None
    return strict, flexible


def answers_match(pred, ref):
    """数值可比时按 float 比较，否则按规范化字符串比较。"""
    if pred is None:
        return False
    try:
        return float(pred) == float(ref)
    except (TypeError, ValueError):
        return normalize_answer(pred) == normalize_answer(ref)


def evaluate(predictions):
    valid = successful(predictions)
    strict_correct = 0
    flexible_correct = 0
    invalid_answer = 0
    for item in valid:
        strict, flexible = extract(item.get("prediction", ""))
        reference = str(item.get("reference", ""))
        if answers_match(strict, reference):
            strict_correct += 1
        if flexible is not None and answers_match(flexible, reference):
            flexible_correct += 1
        if strict is None:
            invalid_answer += 1
    return {
        "num_samples": len(predictions),
        "num_scored": len(valid),
        "num_failed": len(predictions) - len(valid),
        "accuracy": strict_correct / len(valid) if valid else None,
        "strict_acc": strict_correct / len(valid) if valid else None,
        "flexible_acc": flexible_correct / len(valid) if valid else None,
        "strict_correct": strict_correct,
        "flexible_correct": flexible_correct,
        "num_invalid_answer": invalid_answer,
        "metric_note": (
            "accuracy=strict_acc：按‘答案是 X。’(或旧格式)抽取的答案与"
            " reference 数值比较；flexible_acc：答案行截断后最后一个数字；"
            "num_invalid_answer：strict 不可解析的有效样本数。"
        ),
    }
