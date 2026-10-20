import re
from rouge_score import rouge_scorer
from evaluators.common import successful


def char_tokens(text):
    """将文本转为字符序列，以空格分隔，兼容其他评估器的调用。"""
    text = (text or "").lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\w]", "", text, flags=re.UNICODE)
    return " ".join(list(text))


class ChineseCharTokenizer:
    """中文字符级分词器。"""

    def tokenize(self, text):
        return char_tokens(text).split()


def evaluate(predictions):
    valid = successful(predictions)

    if not valid:
        return {
            "num_samples": len(predictions),
            "num_scored": 0,
            "num_failed": len(predictions),
            "rouge1": None,
            "rouge2": None,
            "rougeL": None,
        }

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=ChineseCharTokenizer(),
    )

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for item in valid:
        reference = item.get("reference", "")
        prediction = item.get("prediction") or ""
        scores = scorer.score(reference, prediction)

        for key in totals:
            totals[key] += scores[key].fmeasure

    n = len(valid)

    return {
        "num_samples": len(predictions),
        "num_scored": n,
        "num_failed": len(predictions) - n,
        **{key: value / n for key, value in totals.items()},
    }