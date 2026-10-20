from rouge_score import rouge_scorer
from evaluators.common import successful
from evaluators.summarization import char_tokens


def evaluate(predictions):
    # 注意：ROUGE-L 仅衡量文本重叠，不代表医学正确性或安全性。
    valid = successful(predictions)
    if not valid:
        return {
            "num_samples": len(predictions), "num_scored": 0,
            "num_failed": len(predictions), "rougeL": None,
            "warning": "需增加医学专家或经过验证的医学评估。",
        }

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    total = 0.0
    for item in valid:
        ref = char_tokens(item.get("reference", ""))
        pred = char_tokens(item.get("prediction", ""))
        total += scorer.score(ref, pred)["rougeL"].fmeasure

    return {
        "num_samples": len(predictions),
        "num_scored": len(valid),
        "num_failed": len(predictions) - len(valid),
        "rougeL": total / len(valid),
        "warning": "ROUGE-L 仅为文本重叠参考指标，不代表医学正确性或安全性。",
    }
