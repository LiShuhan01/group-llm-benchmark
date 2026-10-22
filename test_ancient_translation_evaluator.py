from evaluators.ancient_translation import evaluate

rows = [
    {"status": "success", "prediction": "天下大雨。", "reference": "天下大雨。"},
    {"status": "success", "prediction": "百姓安居。", "reference": "百姓安居。"},
    {"status": "failed", "prediction": "", "reference": "参考译文"},
]
result = evaluate(rows)
assert result["num_samples"] == 3
assert result["num_scored"] == 2
assert result["num_failed"] == 1
assert result["bleu"] is not None
print(result)
