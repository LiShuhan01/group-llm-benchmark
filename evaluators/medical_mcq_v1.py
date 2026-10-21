"""XMedBench 医学选择题评估器：支持单选与多选。"""

import re
from collections import defaultdict

from evaluators.common import successful


def normalize_choices(value):
    """将 C、(C)、ACE、(ACE) 等答案规范化为选项字母集合。"""
    if value is None:
        return None

    text = str(value).strip().upper()
    text = re.sub(r"[\s()（）,，、]+", "", text)

    if not text or not re.fullmatch(r"[A-E]+", text):
        return None

    if len(set(text)) != len(text):
        return None

    return "".join(sorted(text))


def extract_choice(text):
    """从模型输出中提取明确的答案，避免从解释正文中随意猜测。"""
    if text is None:
        return None

    text = str(text).strip().upper()

    patterns = [
        r"(?:最终答案|正确答案|答案|选项)\s*(?:是|为|：|:)?\s*[（(]?([A-E]{1,5})[）)]?(?![A-Z])",
        r"^\s*[（(]\s*([A-E]{1,5})\s*[）)]",
        r"^\s*([A-E]{1,5})\s*(?:$|[.。:：、,\s])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return normalize_choices(match.group(1))

    return None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate(predictions):
    successful_predictions = successful(predictions)

    correct = 0
    invalid = 0
    single_count = single_correct = 0
    multi_count = multi_correct = 0

    source_stats = defaultdict(
        lambda: {"count": 0, "correct": 0, "invalid": 0}
    )

    for item in successful_predictions:
        reference = normalize_choices(item.get("reference"))
        prediction = extract_choice(item.get("prediction"))

        if reference is None:
            raise ValueError(
                f"无效的医学选择题标准答案：{item.get('reference')!r}"
            )

        is_correct = prediction == reference
        is_invalid = prediction is None
        is_multi = len(reference) > 1

        correct += int(is_correct)
        invalid += int(is_invalid)

        if is_multi:
            multi_count += 1
            multi_correct += int(is_correct)
        else:
            single_count += 1
            single_correct += int(is_correct)

        metadata = item.get("metadata") or {}
        source = str(metadata.get("source", "unknown"))

        stats = source_stats[source]
        stats["count"] += 1
        stats["correct"] += int(is_correct)
        stats["invalid"] += int(is_invalid)

    per_source = {}
    for source, stats in sorted(source_stats.items()):
        per_source[source] = {
            "count": stats["count"],
            "accuracy": ratio(stats["correct"], stats["count"]),
            "invalid_count": stats["invalid"],
            "invalid_rate": ratio(stats["invalid"], stats["count"]),
        }

    return {
        "num_samples": len(predictions),
        "num_scored": len(successful_predictions),
        "num_failed": len(predictions) - len(successful_predictions),
        "accuracy": ratio(correct, len(successful_predictions)),
        "invalid_count": invalid,
        "invalid_rate": ratio(invalid, len(successful_predictions)),
        "single_choice_count": single_count,
        "single_choice_accuracy": ratio(single_correct, single_count),
        "multiple_choice_count": multi_count,
        "multiple_choice_accuracy": ratio(multi_correct, multi_count),
        "per_source": per_source,
        "metric_note": (
            "多选题采用选项集合完全匹配；"
            "无法解析的答案计为错误，不计入正确数。"
        ),
    }
