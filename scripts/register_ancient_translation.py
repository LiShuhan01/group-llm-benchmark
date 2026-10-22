"""Patch existing Group LLM Benchmark to register ancient_translation.
Run from repository root: python scripts/register_ancient_translation.py
Makes a .bak copy of files before editing. Does not edit models.yaml because each
team member must explicitly add this task to the model config they intend to test.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
evaluate_path = ROOT / "evaluate.py"
config_path = ROOT / "configs" / "benchmark.yaml"

def backup(path):
    bak = path.with_suffix(path.suffix + ".ancient_translation.bak")
    if not bak.exists():
        shutil.copy2(path, bak)

def replace_once(text, old, new, label):
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"无法定位 {label} 的插入位置，请手动检查 {path}")
    return text.replace(old, new, 1)

# Patch evaluator import and registry.
text = evaluate_path.read_text(encoding="utf-8")
if "from evaluators.ancient_translation import evaluate as eval_ancient_translation" not in text:
    backup(evaluate_path)
    import_anchor = "from evaluators.summarization import evaluate as eval_summary\n"
    if import_anchor not in text:
        raise RuntimeError("evaluate.py 中找不到 summarization 导入行，请手动添加 ancient_translation 导入。")
    text = text.replace(
        import_anchor,
        import_anchor + "from evaluators.ancient_translation import evaluate as eval_ancient_translation\n",
        1,
    )
    registry_anchor = '    "text_summarization": eval_summary,\n'
    if registry_anchor not in text:
        raise RuntimeError("evaluate.py 中找不到 text_summarization 注册行，请手动添加 EVALUATORS 映射。")
    text = text.replace(
        registry_anchor,
        registry_anchor + '    "ancient_translation": eval_ancient_translation,\n',
        1,
    )
    evaluate_path.write_text(text, encoding="utf-8")
    print("已更新 evaluate.py")
else:
    print("evaluate.py 已注册 ancient_translation，跳过")

# Patch YAML by text insertion to preserve comments and existing formatting.
yaml_text = config_path.read_text(encoding="utf-8")
if "  ancient_translation:" not in yaml_text:
    backup(config_path)
    anchor = "  math_reasoning:\n"
    if anchor not in yaml_text:
        raise RuntimeError("benchmark.yaml 中找不到 math_reasoning 配置段，请手动添加任务配置。")
    block = (
        "  ancient_translation:\n"
        "    enabled: true\n"
        "    dataset_path: data/ancient_translation/test_v1.jsonl\n"
        "    dataset_version: ancient-translation-v1\n"
        "    evaluator: ancient_translation\n"
    )
    yaml_text = yaml_text.replace(anchor, block + anchor, 1)
    config_path.write_text(yaml_text, encoding="utf-8")
    print("已更新 configs/benchmark.yaml")
else:
    print("benchmark.yaml 已包含 ancient_translation，跳过")

print("注册完成。接下来检查配置，并在 configs/models.yaml 的目标模型 tasks 列表中添加 ancient_translation。")
