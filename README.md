# Group LLM Benchmark

组内多任务大语言模型评估框架，旨在为不同研究领域、不同基座模型和不同微调方法提供统一、可复用的评估流程。

本项目将模型接入、推理生成、任务评估、实验记录和结果汇总等通用流程统一起来。每位组员可以继续在自己的工作目录中保存基座模型和 LoRA 权重，通过个人配置接入公共框架，无需将模型权重上传到 GitHub。

**当前定位：可复用的多任务评估框架。** 正式测试集、任务评估标准及跨领域比较规则仍需由组内共同确定。

## 1. 项目目标

组内成员分别开展不同领域的大模型微调，例如：

- 医疗问答（`medical_qa`）
- 数学推理（`math_reasoning`）
- 文本摘要（`text_summarization`）

各成员使用的基座模型、模型规模、微调方法和训练数据可能不同。因此，本项目采用统一的评估接口和实验记录格式，同时允许不同任务使用各自的测试集与评估器。

本项目主要解决以下问题：

1. 如何使用统一的接口调用不同成员的模型？
2. 如何统一测试样本格式，使不同模型可以复用评估流程？
3. 如何分别评估模型在医疗、数学和文本摘要等领域的表现？
4. 如何比较同一个基座模型微调前后的表现？
5. 如何保存完整的实验记录，使评估结果可追溯、可复现？
6. 如何汇总不同模型的评估结果，形成组内统一的实验报告？

需要注意：统一运行接口不意味着不同任务使用相同的评分标准，也不意味着不同任务的分数可以直接比较。

## 2. 当前支持的功能

| 功能 | 当前实现 |
|---|---|
| 多任务登记 | 医疗问答、数学推理、文本摘要 |
| 本地模型推理 | Transformers |
| LoRA 模型接入 | 通过 PEFT 适配器加载 |
| 服务端模型推理 | vLLM OpenAI-compatible API |
| 统一测试数据格式 | JSONL |
| Prompt 模板 | 各任务独立配置 |
| 任务评估器 | 各任务独立实现 |
| 实验记录 | 保存运行配置、数据哈希、Prompt 哈希及运行状态 |
| 结果汇总 | 汇总多次实验的指标 |
| 基座与微调模型比较 | 根据模型关联关系和数据集内容哈希进行配对 |
| 多次运行统计 | 计算均值、标准差、最小值和最大值 |

以上为项目当前设计与代码实现范围。各任务的评估器仍需检查是否满足正式实验要求，不能仅凭接口存在就认定评分标准已经完善。

## 3. 项目结构

```text
group-llm-benchmark/
├── adapters/
│   ├── __init__.py
│   ├── base.py
│   ├── transformers_adapter.py
│   └── vllm_adapter.py
├── configs/
│   ├── benchmark.yaml
│   ├── models.example.yaml
│   └── models.yaml
├── data/
│   ├── text_summarization/
│   │   └── test.jsonl
│   ├── math_reasoning/
│   │   └── test.jsonl
│   └── medical_qa/
│       └── test.jsonl
├── prompts/
│   ├── text_summarization.txt
│   ├── math_reasoning.txt
│   └── medical_qa.txt
├── evaluators/
│   ├── __init__.py
│   ├── common.py
│   ├── summarization.py
│   ├── math.py
│   └── medical.py
├── scripts/
│   └── validate_data.py
├── outputs/
│   ├── runs/
│   └── comparison/
├── run_benchmark.py
├── evaluate.py
├── compare_results.py
├── requirements.txt
├── .gitignore
└── README.md
```

各部分的职责如下：

| 文件或目录 | 职责 |
|---|---|
| `adapters/` | 统一不同推理后端的模型调用接口 |
| `configs/benchmark.yaml` | 公共任务登记、测试集路径、推理参数和输出设置 |
| `configs/models.example.yaml` | 公共模型配置模板 |
| `configs/models.yaml` | 个人模型配置，填写真实模型路径 |
| `data/` | 各任务测试集 |
| `prompts/` | 各任务的输入提示模板 |
| `evaluators/` | 各任务的指标计算逻辑 |
| `run_benchmark.py` | 加载指定模型并生成预测 |
| `evaluate.py` | 对已有预测结果进行任务评分 |
| `compare_results.py` | 汇总实验记录并比较基座与微调模型 |
| `scripts/validate_data.py` | 检查测试数据的基本格式 |

## 4. 整体工作流程

一次完整的模型评估包含以下步骤：

```text
准备基座模型和微调权重
          |
          v
登记模型并填写个人配置
          |
          v
准备对应任务的测试集
          |
          v
检查数据格式和任务配置
          |
          v
运行 run_benchmark.py
          |
          v
生成 predictions.jsonl
          |
          v
运行 evaluate.py
          |
          v
生成 metrics.json
          |
          v
运行 compare_results.py
          |
          v
生成模型汇总和实验比较报告
```

各步骤相互独立：

- 模型推理负责产生预测结果。
- 任务评估器负责计算该任务的指标。
- 结果汇总工具负责整理实验并进行符合条件的配对比较。

这样可以在不重写整个框架的前提下，为新任务增加专属的 Prompt、数据集和评估器。

## 5. 环境安装

建议使用 Python 3.10 或 3.11，并根据服务器已有的 CUDA、PyTorch、Transformers 和 vLLM 环境选择兼容的依赖版本。

如果服务器已经有可用的 Conda 环境，可以直接使用自己的环境。

例如：

```bash
conda activate llm_lsh

cd /data/lsh/group-llm-benchmark

python --version
python -m pip install -r requirements.txt
```

如果需要独立环境，可以创建新的 Conda 环境：

```bash
conda create -n group_benchmark python=3.10 -y
conda activate group_benchmark

cd /data/lsh/group-llm-benchmark

python -m pip install -r requirements.txt
```

注意事项：

1. 安装依赖前，先确认服务器现有 PyTorch 和 CUDA 环境。
2. 不要为了安装评估工具而随意升级正在使用的训练环境。
3. vLLM 与 Transformers 的版本要求可能不同，可以将 vLLM 服务部署在独立环境中。
4. 使用本地 Transformers 后端时，需要确保模型和适配器能够被当前环境加载。

## 6. 公共配置与个人配置

本项目将公共配置和个人模型配置分开。

### 6.1 公共配置

`configs/benchmark.yaml` 用于登记任务、测试集位置、推理参数和输出设置。

例如：

```yaml
benchmark:
  name: group-llm-benchmark
  version: "0.1.0"

tasks:
  text_summarization:
    enabled: true
    dataset_path: data/text_summarization/test.jsonl
    dataset_version: demo-v0.1
    evaluator: summarization

  math_reasoning:
    enabled: false
    dataset_path: data/math_reasoning/test.jsonl
    dataset_version: pending
    evaluator: math

  medical_qa:
    enabled: false
    dataset_path: data/medical_qa/test.jsonl
    dataset_version: pending
    evaluator: medical

inference:
  temperature: 0.0
  max_new_tokens: 256
  seed: 42

output:
  root_dir: outputs/runs
  save_predictions: true
  save_metrics: true
  save_run_metadata: true

comparison:
  record_base_model_id: true
  record_finetuned_model_id: true
  record_dataset_version: true
  record_inference_config: true
```

以上展示的是项目当前公共配置的主要结构。正式评估时，应根据实际任务配置启用状态、测试集路径和数据集版本，并检查运行代码是否正确使用了对应字段。

### 6.2 个人模型配置

每位组员可以将模型保存在自己的工作目录中，例如：

```text
/home/student_a/models/base_model/
/home/student_a/models/lora_adapter/
```

也可以使用服务器上其他有权限访问的目录。

先复制公共模板：

```bash
cd /data/lsh/group-llm-benchmark

cp configs/models.example.yaml configs/models.yaml
```

然后编辑 `configs/models.yaml`，填写自己的真实路径。

**`configs/models.yaml` 是个人配置，不应提交到公共 GitHub 仓库。**

项目已经通过 `.gitignore` 忽略该文件。公共仓库只保留 `configs/models.example.yaml`。

### 6.3 登记基座模型

示例：

```yaml
models:
  my_base:
    backend: transformers
    role: base
    base_model_id: my_base
    base_model: /your/workspace/base_model
    adapter_path: null
    trust_remote_code: false
    device_map: auto
    tasks:
      - text_summarization
```

字段含义：

- `backend`：推理后端，可选 `transformers` 或 `vllm`。
- `role`：模型角色，基座模型填写 `base`。
- `base_model_id`：用于标识基座模型的逻辑 ID。
- `base_model`：基座模型实际路径。
- `adapter_path`：基座模型不加载 LoRA 时填写 `null`。
- `tasks`：模型登记支持的任务。

### 6.4 登记微调模型

示例：

```yaml
  my_finetuned:
    backend: transformers
    role: finetuned
    base_model_id: my_base
    base_model: /your/workspace/base_model
    adapter_path: /your/workspace/lora_adapter
    trust_remote_code: false
    device_map: auto
    tasks:
      - text_summarization
```

这里的 `base_model_id: my_base` 必须与上面基座模型的配置 ID 一致。

对于 LoRA 微调模型：

- `base_model` 指向训练时使用的基座模型。
- `adapter_path` 指向训练得到的 LoRA 适配器目录。
- 基座模型与微调模型分别登记，便于独立运行和比较。

不同组员可以使用不同架构、不同参数规模的基座模型，不要求所有成员使用同一个基座。

### 6.5 vLLM 模型配置

如果模型通过 vLLM 提供 OpenAI-compatible API，可以采用类似配置：

```yaml
  my_vllm_model:
    backend: vllm
    role: base
    base_model_id: my_vllm_model
    api_base: http://127.0.0.1:8000/v1
    api_key: EMPTY
    served_model_name: YOUR_SERVED_MODEL_NAME
    tasks:
      - text_summarization
```

其中：

- `api_base` 必须指向实际运行的服务地址。
- `served_model_name` 必须与服务端登记的模型名称一致。
- `api_key` 应根据实际服务的鉴权要求填写，不应将真实密钥提交到 GitHub。
- `tasks` 只能登记该服务实际能够完成的任务。

如果服务部署在另一台服务器上，还需要确认网络连通性、访问权限和服务端口。

## 7. 测试集格式

各任务统一使用 JSONL 格式，即每一行是一个独立的 JSON 对象。

例如：

```json
{"id":"example_001","task":"text_summarization","input":"待处理的输入文本","reference":"参考答案","metadata":{"split":"test"}}
```

每条样本至少包含：

| 字段 | 含义 |
|---|---|
| `id` | 测试样本的唯一标识 |
| `task` | 任务名称 |
| `input` | 模型需要处理的输入 |
| `reference` | 评分所需的参考答案 |

`metadata` 为可选字段，可用于记录数据来源、子任务、难度或数据划分等信息。

当前任务名称为：

- `text_summarization`
- `math_reasoning`
- `medical_qa`

测试集路径分别为：

```text
data/text_summarization/test.jsonl
data/math_reasoning/test.jsonl
data/medical_qa/test.jsonl
```

### 7.1 数据检查

执行：

```bash
python scripts/validate_data.py
```

该脚本用于检查当前测试文件的基本格式。

需要注意，格式检查通过不代表数据集已经满足正式实验要求。正式测试集还需要进行内容审核、训练集隔离检查、参考答案核验和使用许可确认。

### 7.2 测试集管理原则

组内应尽量统一以下事项：

1. 数据集的来源和使用许可。
2. 测试集的版本和样本数量。
3. 测试集的输入、参考答案和样本 ID 规范。
4. 测试集与训练数据的隔离要求。
5. 数据集更新后的版本变更和记录方式。

正式测试集应避免在评估过程中被随意修改。

如果同一个测试集文件的内容发生变化，应重新记录其哈希，并将其视为新的数据版本或新的数据内容进行管理。

## 8. Prompt 模板

各任务的 Prompt 模板存放在 `prompts/` 目录中：

```text
prompts/text_summarization.txt
prompts/math_reasoning.txt
prompts/medical_qa.txt
```

Prompt 模板中使用 `{input}` 作为输入占位符，运行程序会将测试样本的 `input` 填入模板。

例如，摘要任务的模板：

```text
请为以下文本撰写准确、简洁的摘要。保留核心信息，不添加原文没有的事实。

原文：
{input}

摘要：
```

数学推理和医疗问答使用各自的模板。

需要注意：

- 同一组基座和微调模型比较时，应使用相同 Prompt。
- Prompt 发生变化时，应保留新的版本或哈希记录。
- 不同任务可以使用不同 Prompt，但应明确记录各任务的输入规范。
- 组内正式评估前，应共同确定 Prompt 是否固定以及是否允许不同模型使用不同模板。

## 9. 运行 Benchmark

### 9.1 小样本测试

建议先运行少量样本，检查模型加载、输入格式和预测生成是否正常。

例如：

```bash
python run_benchmark.py \
  --model my_base \
  --task text_summarization \
  --limit 2
```

再运行微调模型：

```bash
python run_benchmark.py \
  --model my_finetuned \
  --task text_summarization \
  --limit 2
```

其中，`my_base` 和 `my_finetuned` 必须替换为 `configs/models.yaml` 中实际登记的模型 ID。

`--limit` 仅用于限制本次运行的样本数量，适合流程检查。小样本结果不能作为正式模型能力结论。

### 9.2 完整测试集评估

确认小样本推理正常后，运行完整测试集：

```bash
python run_benchmark.py \
  --model my_base \
  --task text_summarization
```

再运行微调模型：

```bash
python run_benchmark.py \
  --model my_finetuned \
  --task text_summarization
```

数学推理和医疗问答使用同一运行入口，只需将模型 ID 和任务名称改为对应配置。

### 9.3 推理输出

每次运行会创建独立的实验目录：

```text
outputs/runs/<run_id>/
├── predictions.jsonl
└── run_config.json
```

主要文件：

**`predictions.jsonl`**

保存逐条预测结果，包括样本 ID、模型 ID、任务、生成文本、运行状态及可用的耗时信息。

**`run_config.json`**

保存实验配置快照及相关元数据，例如：

- 实验 ID 和模型 ID。
- 模型角色与基座模型关联信息。
- 任务名称和推理后端。
- 测试集路径、内容哈希和数据版本。
- Prompt 路径与哈希。
- 推理参数和模型配置。
- 运行时间、成功数、失败数和状态。

具体字段以当前代码实际写入的内容为准。

**`metrics.json`**

该文件由评估程序生成，用于保存当前实验的任务指标和评分统计。

## 10. 评估预测结果

推理完成后，使用 `evaluate.py` 计算任务指标。

例如：

```bash
python evaluate.py \
  --predictions outputs/runs/<run_id>/predictions.jsonl \
  --task text_summarization
```

其中 `<run_id>` 替换为实际的实验目录名。

也可以显式指定测试集：

```bash
python evaluate.py \
  --predictions outputs/runs/<run_id>/predictions.jsonl \
  --task text_summarization \
  --data data/text_summarization/test.jsonl
```

显式指定测试集时，应确保它与生成预测时使用的测试集完全一致。

当前评估入口支持：

- `text_summarization`
- `math_reasoning`
- `medical_qa`

每个任务调用自己的评估器。

### 10.1 文本摘要评估

摘要评估器目前包含 ROUGE 指标，例如 ROUGE-1、ROUGE-2 和 ROUGE-L。

ROUGE 主要反映生成摘要与参考摘要之间的文本重叠程度，不等同于摘要的事实正确性、核心信息覆盖率或整体质量。

如果需要形成正式摘要 Benchmark，还应由组内决定是否加入事实一致性、核心信息覆盖、人工评审或其他经过验证的指标。

### 10.2 数学推理评估

数学任务需要根据具体数据集确定答案提取和判断规则。

正式评估时应明确：

- 是否比较最终答案。
- 是否允许等价表达。
- 数值答案的容差规则。
- 复杂题型如何处理。
- 无法解析的答案如何计数。

这些规则需要与任务数据和评估器保持一致，不能仅凭程序运行成功就认为数学评分准确。

### 10.3 医疗问答评估

医疗问答的正确性、安全性和临床适用性不能仅凭文本重叠指标判断。

正式医疗评估需要根据题型和使用场景确定准确率、专家审核或经过验证的评审流程。

涉及医疗结论时，应避免将未经验证的自动评分结果解释为临床有效性证明。

## 11. 汇总实验和比较模型

当各实验已经生成 `metrics.json` 后，可以运行：

```bash
python compare_results.py
```

默认读取：

```text
outputs/runs/
```

默认输出：

```text
outputs/comparison/
├── comparison.csv
├── model_summary.csv
└── report.md
```

### 11.1 comparison.csv

这是逐次实验明细表，记录每次运行的模型、任务、指标、得分、数据集哈希及可配对的基座得分和差值。

适合追踪每一次实验及其对应的比较结果。

### 11.2 model_summary.csv

这是模型指标的多次运行汇总表，通常包含：

- 模型 ID。
- 模型角色。
- 基座模型 ID。
- 任务和指标。
- 数据集版本及内容哈希。
- 运行次数。
- 均值、标准差、最小值和最大值。

这些统计量有助于观察多次实验的表现及稳定性，但样本数量和运行次数不足时，不能据此得出可靠的统计显著性结论。

### 11.3 report.md

这是便于阅读的 Markdown 报告，用于查看模型指标汇总及基座—微调模型的配对比较结果。

### 11.4 基座与微调模型配对规则

框架通过以下信息进行关联：

1. 微调模型的 `base_model_id` 与基座模型 ID 对应。
2. 任务名称一致。
3. 测试集内容哈希一致。
4. 指标名称一致。
5. 模型角色和实验元数据正确。

只有满足对应条件的指标才应进行比较。

差值定义为：

```text
差值 = 微调模型得分 - 基座模型得分
```

相对变化率定义为：

```text
相对变化率 = 差值 / abs(基座模型得分) × 100%
```

当基座模型得分为零时，不计算相对变化率。

**差值的正负不一定代表模型变好或变差。** 必须根据指标的含义判断方向。例如，准确率或 ROUGE 通常越高越好，而推理耗时通常越低越好。

如果配对结果缺失，应检查模型角色、`base_model_id`、数据集哈希及指标名称，不要仅根据模型显示名称判断是否配对成功。

## 12. 如何开展跨领域能力评估

组内的研究目标不仅是评价每个模型在自己的研究领域表现如何，还希望了解微调是否提升目标任务能力，以及是否影响其他领域的能力。

需要区分两种实验。

### 12.1 目标领域微调效果

例如，某位组员针对数学推理进行微调。

应分别运行：

- 数学基座模型在数学测试集上的评估。
- 数学微调模型在同一数学测试集上的评估。

在测试集、Prompt、推理设置和评估器一致的前提下，比较两者的指标差异。

这能够为微调是否改善目标任务表现提供证据。

同理，摘要微调模型应与其对应基座模型在同一摘要测试集上比较。

### 12.2 跨领域能力评估

如果希望知道一个模型在医疗、数学和文本摘要领域分别表现如何，就需要让该模型的基座版本和微调版本运行所有适用的测试任务。

例如：

| 模型版本 | 医疗问答 | 数学推理 | 文本摘要 |
|---|---|---|---|
| 基座模型 | 评估 | 评估 | 评估 |
| 微调模型 | 评估 | 评估 | 评估 |

每个单元格都应使用对应任务的测试集和评估器。

这类实验可以用于观察：

- 目标领域的能力是否提升。
- 其他领域的表现是否发生变化。
- 微调是否存在跨领域能力迁移或能力退化。

但不同模型的基座、规模、训练数据和微调方法可能不同，因此不能将所有模型之间的原始分数差异直接归因于微调。

如果不同模型在某个任务上采用不同测试集或不同评分规则，也不应直接比较原始分数。

### 12.3 不要直接合并不同任务的分数

例如，数学准确率、摘要 ROUGE 和医疗评审得分的含义与量纲可能完全不同。

因此，不应直接将它们求和或取平均形成总分，除非组内明确制定了经过论证的归一化、权重及汇总规则。

在正式方案确定之前，应分别展示各任务指标，并明确数据集和评分方法。

## 13. 新组员如何接入

每位组员可以独立完成以下步骤。

### 第一步：获取公共代码

```bash
git clone https://github.com/LiShuhan01/group-llm-benchmark.git

cd group-llm-benchmark
```

如果已经克隆过仓库，则进入对应目录并按团队约定更新公共代码。

### 第二步：准备模型

继续使用自己工作目录中的基座模型和 LoRA 权重，不必复制到公共项目目录。

### 第三步：创建个人配置

```bash
cp configs/models.example.yaml configs/models.yaml
```

编辑个人配置，填写模型路径，并为基座和微调模型登记正确的 ID、角色及任务。

### 第四步：准备测试集

按照本项目的 JSONL 格式准备对应任务的测试数据，并检查数据内容和参考答案。

### 第五步：确认任务配置

检查 `configs/benchmark.yaml` 中对应任务是否启用、数据路径是否正确、数据版本是否明确。

同时确认 Prompt 模板和任务评估器适用于自己的研究任务。

### 第六步：运行少量样本

```bash
python run_benchmark.py \
  --model YOUR_MODEL_ID \
  --task YOUR_TASK \
  --limit 2
```

替换模型 ID 和任务名称后执行。

### 第七步：运行完整评估

```bash
python run_benchmark.py \
  --model YOUR_MODEL_ID \
  --task YOUR_TASK
```

### 第八步：计算指标

```bash
python evaluate.py \
  --predictions outputs/runs/YOUR_RUN_ID/predictions.jsonl \
  --task YOUR_TASK
```

### 第九步：汇总实验

```bash
python compare_results.py
```

检查结果是否正确配对，并确认模型角色、数据集哈希和评分指标没有错误。

### 第十步：保存结果

保留实验目录和汇总报告，以便后续复查、组会汇报和实验复现。

## 14. 新增任务需要修改什么

通用框架尽量复用已有代码，但不同任务仍然需要任务专属实现。

如果组内增加新任务，通常需要完成以下工作：

1. 确定任务 ID 和数据字段规范。
2. 准备测试集并登记数据版本。
3. 编写对应的 Prompt 模板。
4. 实现任务专属评估器。
5. 将任务加入公共配置。
6. 根据需要扩展 `evaluate.py` 的任务分发逻辑。
7. 使用少量样本检查推理和评估流程。
8. 检查结果记录与比较逻辑是否正确处理新任务的指标。

如果已有通用模型适配器能够完成该任务，通常不需要为每个新任务单独编写推理后端。

但如果新任务需要特殊的输入处理、生成参数或模型调用方式，就需要扩展对应的通用接口。

## 15. 组内需要共同确定的事项

公共运行框架已经提供了统一接口的基础，但正式跨领域 Benchmark 还需要组内共同确定以下内容。

### 15.1 测试集

- 各任务正式使用哪个数据集。
- 测试集来源、使用许可和审核流程。
- 测试集规模及样本筛选规则。
- 测试集版本如何管理。
- 如何确保测试数据与训练数据隔离。

### 15.2 评估标准

- 摘要任务使用哪些指标。
- 数学任务如何判断答案正确。
- 医疗任务如何评价正确性和安全性。
- 是否需要人工审核或模型评审。
- 失败样本、无法解析答案和缺失输出如何计分。

### 15.3 运行规范

- 是否固定 Prompt。
- 是否统一温度、最大生成长度等参数。
- 是否需要多次重复运行。
- 是否记录推理延迟和资源消耗。
- 是否需要统一随机种子。

### 15.4 跨领域比较规则

- 哪些模型需要参加哪些任务。
- 是否要求每个模型都运行全部领域。
- 如何区分目标任务提升与跨领域能力变化。
- 是否需要构建综合分数。
- 不同模型之间的比较应如何解释。

这些问题需要结合实际研究目标共同决定，不能仅通过统一代码自动解决。

## 16. GitHub 协作与文件管理

公共仓库用于保存所有组员都需要的通用内容。

适合提交的内容：

- 通用推理和评估代码。
- 公共配置模板。
- 经过许可的数据集。
- Prompt 模板和评估规则。
- 文档、依赖说明和测试代码。

不应提交的内容：

- 个人模型配置 `configs/models.yaml`。
- 大型基座模型权重和 LoRA 权重。
- API 密钥、访问令牌及其他凭证。
- 未经许可的数据集。
- 含有敏感信息的测试数据。
- 临时备份文件。
- 不需要共享的个人实验输出。

本项目的 `.gitignore` 已包含对个人配置、输出目录和部分模型文件的忽略规则。提交前仍应检查 Git 实际状态，不要仅凭忽略规则就假定所有个人文件都不会被提交。

### 16.1 提交前检查

```bash
git status --short
git diff --check
git diff --stat
```

确认需要提交的文件后，再逐项暂存。

例如：

```bash
git add README.md
git add configs/models.example.yaml
git add configs/benchmark.yaml
git add run_benchmark.py evaluate.py compare_results.py
git add adapters evaluators scripts requirements.txt prompts
```

只有在确认数据文件适合共享后，才将相关数据加入暂存区。

随后检查：

```bash
git status
git diff --cached --check
git diff --cached --stat
```

确认暂存区没有个人配置、模型权重、密钥或临时文件后，再提交：

```bash
git commit -m "Improve benchmark framework and documentation"
git push
```

## 17. 结果解释与实验注意事项

正式汇报时，建议至少报告以下信息：

- 模型名称和基座模型。
- 微调方法及适配器信息。
- 任务名称。
- 测试集版本、样本数量及内容哈希。
- Prompt 和生成配置。
- 使用的评估指标。
- 成功、失败和未执行样本数量。
- 基座模型和微调模型的得分及差值。
- 重复运行的统计结果（如果进行了重复实验）。

不要仅根据一次小样本测试得出微调有效或无效的结论。

同样，某个微调模型在目标领域得分提高，并不意味着它在所有领域都变强了。跨领域能力需要独立评估。

---

## 当前版本说明

Group LLM Benchmark 的目标是为组内不同研究领域、不同基座模型和不同微调模型提供可复用的评估基础设施。

目前的核心思路是：

- 统一模型调用入口。
- 统一测试数据的基本格式。
- 保留各任务独立的 Prompt 和评估器。
- 统一实验元数据与结果存储。
- 支持基座与微调模型的配对比较。
- 支持多次实验结果汇总。
- 将模型权重和个人路径留在各自工作目录中。

**当前项目仍处于框架建设阶段。** 正式测试集、任务指标和跨领域比较方案需要由组内共同确定。只有在这些内容经过审查并固定后，才能将评估结果作为正式的模型能力结论。

## LCSTS 与 XMedBench 评估数据

本项目目前准备了三个评估任务：

| 任务 | 数据集 | 当前样本数 | 评估指标 |
|---|---|---:|---|
| `text_summarization` | LCSTS | 1,000 | ROUGE-1、ROUGE-2、ROUGE-L |
| `medical_mcq_xmedbench_v1` | XMedBench 中文集 | 8,759 | 准确率、单选/多选准确率、无效答案率 |
| `math_reasoning` | MGSM zh | 250 | strict_acc、flexible_acc（无效答案另计） |

注意：当前 LCSTS 评估样本来自 validation 分割的固定随机种子抽样，并非独立官方测试集。医疗选择题准确率只衡量答案与参考答案的匹配情况，不代表完整的医学安全性评估。

### 1. 准备 LCSTS 数据

先从获准的数据来源获取 LCSTS，并确保数据已使用 Hugging Face Datasets 的 `save_to_disk` 格式保存。

```bash
python scripts/prepare_lcsts.py \
  --source /你的实际路径/LCSTS \
  --size 1000 \
  --seed 42
```

### 2. 准备 MGSM zh 数学数据

`math_reasoning` 已正式接入 MGSM（GSM8K 的人工翻译多语言版）中文 test 集：
250 题，答案为数值（`answer_number`），无空字段、id 唯一。

- 数据文件：`data/math_reasoning/mgsm_zh_test_v1.jsonl`
- 数据版本：`mgsm-zh-test-v1`（`configs/benchmark.yaml` 已启用，任务源 juletxara/mgsm 配置 zh、split test，全量使用、不做抽样）
- 参考文件哈希（sha256，写入实验口径时以此为准）：
  `357fa42702f2fe626db986f3fc8892e0251d6496d50738e138e1e87a75fad8fb`

转换命令（二选一的数据源）：

```bash
# 优先：显式指定本地 parquet / jsonl 源（离线、可追溯）
python scripts/prepare_math_mgsm.py --source /你的实际路径/mgsm_zh_test.parquet

# 自动探测：同机器上存在 ../evalweave/data/mgsm_zh/test_v1.jsonl 时直接复用；
# 两者皆缺时脚本会从 HF 镜像下载官方 parquet（需联网）。
python scripts/prepare_math_mgsm.py
```

生成后同一来源与行序的样本 id 与 `benchmarks/evalweave` 的 MGSM 数据完全一致，
两套评测流程可以逐题对照。

#### Prompt 口径

`prompts/math_reasoning.txt` 为 8-shot 思维链模板：MGSM train 前 8 题范例
烘焙在模板内，末尾是 `问题: {input}\n逐步解答:`。范例文本派生自
benchmarks/evalweave 已逐题对照验证的 `mgsm_zh_cot_v1` 模板（仅将占位符
`{question}` 换成 `{input}`、去掉 ChatML 包裹），Prompt 哈希：

sha256 `9519f0b0711e1d211c66edfae87fa9707da8c20f1b264cbd9725903b020acd9c`

注意：Transformers 适配器会用 `apply_chat_template` 包裹 user 消息
（Qwen 模板自动补默认 system 头），因此 **模板文件内不要再写
`<|im_start|>` 等 ChatML 标记**，否则会被双重包裹。

#### 评估口径（strict / flexible 双口径）

math 评估器在解析前先做两处截断：复述的下一个"问题:"处、以及首个完整答案行
（`#### N` 或 `答案是 N。`）处，复读内容不参与抽取，然后：

- `strict_acc`（= `accuracy`）：按"答案是 X。"正则抽取（旧格式
  "最终答案：/答案：/\\boxed{...}" 作为回退保留），与 reference 按数值比较
  （均可解析为 float 时，如 40 == 40.0）；
- `flexible_acc`：答案行截断后最后一个数字，与 reference 数值比较；
- `num_invalid_answer`：strict 抽不到答案的有效样本数（此类样本计入
  flexible 的分母，仍可命中 flexible_acc）。

该口径与 `qwen_sft/eval_mgsm.py` 及 benchmarks/evalweave 的
`MGSMZhCotTask` 一致，评估行为由 `scripts/test_math_evaluator.py`
的 10 组边界用例锁定（正确/错误/空输出/复读截断/数值等价/旧格式兼容等）。

#### 运行命令

```bash
# 冒烟（先各 3 条，确认 prediction 里出现"答案是 X。"或可解析数字）
python run_benchmark.py --model qwen25_3b_base_math --task math_reasoning --limit 3
python run_benchmark.py --model qwen25_3b_qlora_math3b --task math_reasoning --limit 3

# 全量 250 题
python run_benchmark.py --model qwen25_3b_base_math --task math_reasoning
python run_benchmark.py --model qwen25_3b_qlora_math3b --task math_reasoning

# 评分
python evaluate.py --predictions outputs/runs/<run_id>/predictions.jsonl --task math_reasoning

# 评估器回归
python scripts/test_math_evaluator.py
```

个人模型配置（`configs/models.yaml`，不入库）示例中登记了
`qwen25_3b_base_math`（Qwen2.5-3B 基座）与
`qwen25_3b_qlora_math3b`（同一基座 + math3b QLoRA 适配器），
`tasks` 均为 `math_reasoning`；路径与 benchmarks/evalweave 的模型配置一致，
便于跨框架逐题核对。

#### 已知限制

- 与 evalweave 流程的差异：本框架由适配器统一包 ChatML，且无自定义
  stopping criteria，基座模型多数题会生成满 256 token，速度偏慢属正常，
  判分端截断兜底（两框架行为等价）；
- MGSM 为 GSM8K 的翻译数据，中文译文可能存在少量翻译噪声；
  许可以 juletxara/mgsm 原数据集卡片为准；
- `data/math_reasoning/test.jsonl`（2 条 demo 乘法题）保留作为格式示例，
  正式评估一律使用 `mgsm_zh_test_v1.jsonl`；
- 结果比较需以束、Prompt、生成参数一致为前提；小样本（--limit）结果
  不能作为模型能力结论。
