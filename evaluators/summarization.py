from rouge_score import rouge_scorer
from evaluators.common import successful


def char_tokens(text):
    # 字符级近似；正式评估请固定并记录中文分词方案。
    return " ".join(list((text or "").replace(" ", "")))


def evaluate(predictions):
    valid = successful(predictions)
    if not valid:
        return {
            "num_samples": len(predictions), "num_scored": 0,
            "num_failed": len(predictions), "rouge1": None,
            "rouge2": None, "rougeL": None,
        }

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for item in valid:
        ref = char_tokens(item.get("reference", ""))
        pred = char_tokens(item.get("prediction", ""))
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure

    n = len(valid)
    return {
        "num_samples": len(predictions),
        "num_scored": n,
        "num_failed": len(predictions) - n,
        **{key: value / n for key, value in totals.items()},
    }
