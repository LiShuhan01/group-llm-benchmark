"""BLEU evaluator for classical-Chinese to modern-Chinese translation."""
from sacrebleu.metrics import BLEU
from evaluators.common import successful


def evaluate(predictions):
    valid = successful(predictions)
    if not valid:
        return {
            "num_samples": len(predictions),
            "num_scored": 0,
            "num_failed": len(predictions),
            "bleu": None,
            "metric_note": "SacreBLEU BLEU(tokenize='zh'); no successful predictions to score.",
        }

    hypotheses = [str(item.get("prediction") or "").strip() for item in valid]
    references = [str(item.get("reference") or "").strip() for item in valid]
    scorer = BLEU(tokenize="zh")
    score = scorer.corpus_score(hypotheses, [references]).score

    return {
        "num_samples": len(predictions),
        "num_scored": len(valid),
        "num_failed": len(predictions) - len(valid),
        "bleu": score,
        "metric_note": (
            "SacreBLEU corpus BLEU with tokenize='zh', matching the provided "
            "evaluation script. BLEU measures reference overlap and does not "
            "fully capture translation adequacy or factual fidelity."
        ),
    }
