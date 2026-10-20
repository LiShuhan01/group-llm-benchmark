# Group LLM Benchmark

组内多任务大模型评估项目第一版，支持：
- `medical_qa`：医疗问答
- `math_reasoning`：数学推理
- `text_summarization`：文本摘要
- Transformers + PEFT/LoRA 本地推理
- vLLM OpenAI-compatible API 推理

## 重要说明
`data/*/test.jsonl` 内是流程演示样例，不是正式 benchmark 数据。正式评估前请替换为经过审核、与训练集隔离且有使用许可的测试集。

医疗评估器目前使用 ROUGE-L 作为文本重叠参考指标，不能代表医学正确性或安全性。正式医疗评估需根据题型加入准确率、专家审核或经验证的评审流程。

## 安装
建议 Python 3.10 或 3.11。先创建独立环境，再根据服务器 CUDA/PyTorch 环境安装依赖。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若服务器未安装 `venv`，请联系管理员或使用已有 conda 环境。vLLM 与 PyTorch/Transformers 版本可能有兼容要求；vLLM 服务可放在独立环境运行，本项目客户端只依赖 `requests`。

## 配置模型
编辑 `configs/models.yaml`，将 `/YOUR/PATH/...` 替换为服务器上的真实路径：
- `base_model`：基座模型本地目录或可访问的模型 ID
- `adapter_path`：LoRA 权重目录；评估基座模型时使用 `null`
- vLLM 配置的 `api_base`、`served_model_name` 必须与服务端一致

## 检查数据
```bash
python scripts/validate_data.py
```

每行一个 JSON 对象，至少包含 `id`、`task`、`input`、`reference`。

## 运行
```bash
python run_benchmark.py --model qwen_summary_lora --task text_summarization --limit 1
```

对 vLLM 模型：
```bash
python run_benchmark.py --model qwen_base_vllm --task math_reasoning --limit 2
```

运行输出在 `outputs/runs/<run_id>/`：
- `predictions.jsonl`：逐条预测
- `run_config.json`：本次配置快照和数据/Prompt 哈希
- `metrics.json`：评估指标

单独重算指标：
```bash
python evaluate.py --predictions outputs/runs/<run_id>/predictions.jsonl --task text_summarization
```

## vLLM 服务示例
在另一终端、已安装且版本兼容的 vLLM 环境中运行，先用 `vllm serve --help` 检查参数：

```bash
vllm serve /YOUR/PATH/TO/BASE_MODEL --served-model-name qwen-base --host 127.0.0.1 --port 8000
```

若要服务 LoRA，需确认当前 vLLM 版本支持对应 LoRA 参数。不要将推理服务暴露给不可信网络。

## 公平性与复现
固定测试集版本、Prompt、生成参数和评分代码；同时报告成功/失败样本数。比较微调模型与基座模型时，应使用同一任务、同一测试集和同一生成配置。不要直接平均数学准确率、ROUGE 和医疗评分等不同量纲的指标。
