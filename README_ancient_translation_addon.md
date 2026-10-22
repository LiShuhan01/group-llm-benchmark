# 古文翻译任务接入补丁

本补丁将用户提供的 JSONL 数据接入 Group LLM Benchmark。

## 已转换的数据
- 文件：`data/ancient_translation/test_v1.jsonl`
- 样本数：19981
- 原始字段：`id`, `task`, `input`, `reference`
- 转换后增加 `metadata`，并统一 `task` 为 `ancient_translation`
- BLEU 标准：`sacrebleu.metrics.BLEU(tokenize="zh")`，与用户提供脚本保持一致

## 复制到现有仓库
把本补丁压缩包中的 `data/`、`evaluators/`、`prompts/`、`scripts/` 目录合并复制到 `/data/lsh/group-llm-benchmark/`，不要覆盖原有目录中的其他文件。

然后在仓库根目录执行：
```bash
pip install sacrebleu
python scripts/register_ancient_translation.py
python -m py_compile evaluate.py evaluators/ancient_translation.py scripts/prepare_ancient_translation.py
python evaluate.py --help
```

## 配置模型
在 `configs/models.yaml` 的目标模型配置 `tasks` 列表中添加 `ancient_translation`。模型基座与 LoRA 路径必须填写你自己的真实路径；不要照搬用户脚本中的 `/data/jl/llm-ft/...` 路径。

## 小样本运行
```bash
python run_benchmark.py --model 你的模型配置名 --task ancient_translation --limit 3
python evaluate.py --predictions outputs/runs/实际运行目录/predictions.jsonl --task ancient_translation
```
确认无误后，去掉 `--limit 3` 跑完整测试集。

## 指标解释
`bleu` 是 SacreBLEU 的 corpus BLEU 分数，采用中文分词模式。它衡量模型译文与参考译文的 n-gram 重叠，并不能完整衡量语义等价、漏译、错译或译文流畅度。正式报告中应同时报告测试集版本和样本数量。

## 数据来源与共享提醒
上传前请确认原始数据集的许可证及公开再分发条件。数据文件虽已随补丁打包，不代表其上游许可一定允许公开分发。
