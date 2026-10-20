import re
from evaluators.common import successful


def extract_final_answer(text):
    if not text:
        return ""
    patterns = [
        r"最终答案\s*[:：]\s*(.+)",
        r"(?:答案|Answer)\s*[:：]\s*(.+)",
        r"\\boxed\{([^{}]+)\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().splitlines()[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize(text):
    text = (text or "").strip().lower()
    text = text.replace(",", "").replace("，", "")
    text = re.sub(r"\s+", "", text)
    return text.rstrip("。.!！")


def evaluate(predictions):
    valid = successful(predictions)
    correct = 0
    for item in valid:
        pred = normalize(extract_final_answer(item.get("prediction", "")))
        ref = normalize(item.get("reference", ""))
        if pred == ref:
            correct += 1
    return {
        "num_samples": len(predictions),
        "num_scored": len(valid),
        "num_failed": len(predictions) - len(valid),
        "accuracy": correct / len(valid) if valid else None,
        "correct": correct,
    }
