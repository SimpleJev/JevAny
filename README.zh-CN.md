<p align="center">
  <img src="docs/title.png" alt="JevAny：从任意模型构建你的 Jev，部署到任意应用" width="100%">
</p>

<p align="center">
  <a href="https://simplejev.github.io/JevAny/"><img alt="项目主页" src="https://img.shields.io/badge/website-JevAny-2dd4bf"></a>
  <a href="https://huggingface.co/collections/SimpleJev/jevany-adaptive-decision-systems-6abc8fd39266b1c17b11b4e6"><img alt="模型" src="https://img.shields.io/badge/%F0%9F%A4%97-checkpoints-ffb000"></a>
  <a href="docs/API.md"><img alt="API 文档" src="https://img.shields.io/badge/docs-API-0ea5e9"></a>
  <a href="docs/CASES.md"><img alt="示例" src="https://img.shields.io/badge/examples-gallery-8b5cf6"></a>
  <a href="pyproject.toml"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&amp;logoColor=white"></a>
  <a href="https://github.com/SimpleJev/JevAny/actions"><img alt="测试" src="https://github.com/SimpleJev/JevAny/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="许可证" src="https://img.shields.io/badge/license-Apache--2.0-32d6c5"></a>
</p>

<p align="center">
  <a href="README.md">🇺🇸 English</a> | <strong>🇨🇳 简体中文</strong><br>
  <strong><a href="#结果与演示">🎮 结果与演示</a> |
  <a href="#快速上手">⚡ 快速上手</a> |
  <a href="#本地体验">💻 本地体验</a> |
  <a href="#预训练模型">🤗 模型</a> |
  <a href="#评测">📊 评测</a> |
  <a href="#文档与贡献">📚 文档</a></strong>
</p>

**JevAny 是面向 System 1 决策模型训练与部署的开源 infra**，涵盖数据准备、模型适配和评测。
你可以直接使用已发布模型，也可以用自己的数据训练，用于工单分流、工具选择和机器人动作决策。
统一 API 接收状态、问题和候选选项，直接返回选择结果与各选项概率。

<p align="center">
  <img src="docs/hero.png" alt="JevAny infra：System 1 决策模型训练、部署与应用集成" width="100%">
</p>

## 🎮 结果与演示 <a name="结果与演示"></a><a name="演示"></a>

[![左右并列柱状图：JevAny checkpoint 与基线模型在 Transfer 和 JevBench 上的准确率](docs/evaluation-summary.svg)](#评测)

[完整基准测试结果与评测说明](#评测)。

以下 30 个案例是由早期兼容 checkpoint 录制的历史回放，展示了 JevAny 在机器人、浏览器、软件、实验室和出行任务中的动作选择；
当前默认发布模型为
[JevAny-Qwen3.8-27B](https://huggingface.co/SimpleJev/JevAny-Qwen3.8-27B-LoRA)。[查看全部案例](docs/CASES.md)，
或[在本地运行模型](#本地体验)，输入自己的任务，查看模型的选择和各选项概率。

[![JevAny 在机器人、浏览器、软件、实验室和出行任务中选择动作](docs/demos/jevany-cases.gif)](docs/CASES.md)

### ⚡ Jev 加入 LLM Agent 循环

> **关键结论：**LLM 负责规划，环境或 LLM 提供少量选项，Jev 选择常规动作，LLM 验证并完成任务。

**1 · Terminal-Bench · `sqlite-db-truncate`——从三个命令中选择一个**

[![Jev 从三个 Terminal-Bench 命令中选择原始页面检查，LLM 随后解析并验证](docs/demos/jev-agent-harness-sqlite.gif)](reports/JevAny_Tech_Report_Agent_Harness_Appendix.md#h1-sqlite-recovery-a-meaningful-three-way-decision)

**2 · WebShop——从页面动作中选择任务要求的颜色**

[![Jev 选择任务要求的黑色商品选项，LLM 随后选择 11.5 尺码并完成购买](docs/demos/jev-agent-harness-webshop.gif)](docs/demos/jev-agent-harness-traces.json)

**3 · FrozenLake——每一步都比较四个方向**

[![LLM 规划一次后，Jev 在 FrozenLake 中连续选择四个依赖当前状态的导航动作](docs/demos/jev-agent-harness-frozen-lake.gif)](results/agent-harness-v1/formal-matrix.md)

| 任务 | 成功率 | 效率 |
|---|---:|---:|
| GPT-5.6-sol · FrozenLake · 10 pairs | 100% → 100% | LLM calls −64.4% · tokens −63.1% · 时间 −37.6% |
| WebShop · 10 pairs | 50% → 60% | LLM calls −7.7% · tokens −2.9% · 时间 −6.1% |
| WebArena · 6 pairs | 50% → 50% | LLM calls +5.6% · tokens +28.2% · 时间 −0.4% |
| Terminal-Bench · 6 pairs | 1/6 → 3/6 | LLM calls −9.0% |

**适合交给 Jev：**2–4 个有边界、可回退、能立即观察结果的选择。

**保留给 LLM：**规划、精确修改、失败恢复和最终答案。

[完整结果](reports/JevAny_Tech_Report_Agent_Harness_Appendix.md) ·
[组合动画](docs/demos/jev-agent-harness.gif) ·
[任务与委托分级](docs/experiments/AGENT_HARNESS_FRONTIER_PROTOCOL.md) ·
[合并版技术报告](reports/JevAny_Tech_Report_with_Agent_Harness.pdf)

## 📑 目录

- [🎮 结果与演示](#结果与演示)
- [⚡ 1. 快速上手](#快速上手)
  - [💻 1.1 本地体验](#本地体验)
  - [🛠️ 1.2 JevAny 训练](#训练)
  - [🚀 1.3 JevAny 部署](#部署)
- [🤗 2. 预训练模型](#预训练模型)
- [📊 3. 基准测试结果](#评测)
- [🕹️ 4. 示例与测试环境](#示例与测试环境)
- [🧩 5. 支持的模型系列](#支持的模型系列)
- [📚 6. 文档与贡献](#文档与贡献)

## ⚡ 1. 快速上手 <a name="快速上手"></a>

使用 Python 3.12 或更新版本，克隆仓库并安装轻量客户端：

```bash
git clone https://github.com/SimpleJev/JevAny.git
cd JevAny
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

以下命令均在仓库根目录运行，并使用上述虚拟环境。
先在本地体验，再用自己的数据[训练模型](#训练)，或通过 [API 接入应用](#部署)。

### 💻 1.1 本地体验 <a name="本地体验"></a>

选择适合自己电脑的模型：

| 模型 | 硬件 | 从这里开始 |
|---|---|---|
| Qwen 0.8B 入门配置 | CPU · 建议 16 GB 内存 | 用随包工单[训练小型 adapter](docs/TRAINING.md#start-with-a-small-backbone) |
| JevAny-Qwen 4B | CUDA · BF16 基座权重约 8 GB，另需运行时显存 | [加载已发布模型](#部署) |
| JevAny-Qwen 27B | CUDA · BF16 基座权重约 54 GB，另需运行时显存 | [选择更大的 checkpoint](docs/PLAYGROUND.md#choose-and-load-a-model) |

准备和加载步骤见[本地模型指南](docs/PLAYGROUND.md)。已发布模型首次使用时下载，
之后复用本地缓存。保持模型服务运行，在同一仓库目录打开第二个终端：

```bash
source .venv/bin/activate
jevany demo --base-url http://127.0.0.1:8008 --text-only
```

打开 `http://127.0.0.1:8090`，点击 **Test and connect**，在
**Try your own decision** 中输入任务，点击 **Ask the model**。
修改状态或候选选项，观察模型的选择如何变化。
同一界面还提供[游戏、机器人和回放](#打开交互演示)。

### 🛠️ 1.2 JevAny 训练 <a name="训练"></a>

用标注决策数据训练自己的 System 1 模型：数据沿用推理时的 `state` 和 `questions`，
为每个问题增加标签。
先用随包提供的合成客服工单开始训练，再换成自己的标注数据。入门配置在 CUDA 上
以 BF16 训练 Qwen3.5-0.8B，结果写入 `runs/my-jev`：

```bash
python -m pip install -e '.[train]'
jevany data init --out data/starter
jevany data validate data/starter/train.jsonl
jevany train --config recipes/sft.toml --dry-run
jevany train --config recipes/sft.toml
```

训练完成后，用随包提供的工单请求试用模型：

```bash
jevany decide examples/request.json --checkpoint runs/my-jev
```

通过 `--data` 指定自己的 [JSONL 数据](docs/DATA.md)，或用
[`recipes/finetune.toml`](recipes/finetune.toml) 微调已发布的 27B 模型。
CPU 配置、多模态数据和标准 `torchrun` 启动方式见[训练指南](docs/TRAINING.md)。
训练图片/视频模型或微调已发布的 27B 模型时，安装 `.[train,multimodal]`。

完成 SFT 后，可用实验性的 [RLCR](docs/ALGORITHM.md#rlcr) 继续训练，其奖励兼顾正确率与概率校准：

```bash
jevany train --config recipes/rlcr.toml
```

### 🚀 1.3 JevAny 部署 <a name="部署"></a>

安装推理依赖，在 CUDA GPU 上启动已发布的 Qwen 4B 模型。
显存要求见[硬件与加载说明](docs/DEPLOYMENT.md#checkpoints-and-hardware)。

```bash
python -m pip install -e '.[serve,multimodal]'
jevany serve --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA \
  --device cuda --dtype bf16 --port 8008
```

默认路径优先保证结果可复现。CUDA 部署可选择 BF16 LoRA 融合、SDPA 和
`torch.compile`，4B 与 27B 的推荐配置不同。具体命令、H200 实测数据和精度说明见
[推理加速指南](docs/DEPLOYMENT.md#optional-cuda-acceleration)。

部署自己的训练结果时，将 checkpoint ID 替换为 `runs/my-jev`。
保持服务运行，在使用相同虚拟环境的 Python 会话中，发送工单和候选处理部门：

```python
from jevany import Choice, JevClient

jev = JevClient("http://127.0.0.1:8008")
result = jev.system_one(
    state={"ticket": "I was charged twice. Please help."},
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Payment problems", "shipping": "Delivery problems"},
        ),
    },
)
answer = result["answers"]["department"]
print("Selected team:", answer["choice"])
print("Probabilities:", answer["probabilities"])
```

`choice` 返回一个候选部门名称，`probabilities` 返回各部门的概率。
你可以据此分配工单，也可以在结果不确定时转交人工审核。
二分类问题使用 `Noul`，例如判断工单是否需要紧急处理；有序评分使用 `Score`，
例如低、普通、高三个优先级。三类问题的完整格式见 [API 文档](docs/API.md)。

进程内推理可以[在 Python 中加载模型](docs/DEPLOYMENT.md#python)，通过相同接口调用。
图片和视频输入见[媒体配置](docs/DEPLOYMENT.md#native-media-and-limits)。

## 🤗 2. 预训练模型 <a name="预训练模型"></a>

第一次在本地运行，可以先按[本地体验](#本地体验)中的硬件要求选择模型。

| 模型 | Readout | 用途 |
|---|---|---|
| [<img src="docs/model-logos/jevany-gemma.svg" width="24" height="24" align="middle" alt="">&nbsp;JevAny-Gemma-4B](https://huggingface.co/SimpleJev/JevAny-Gemma-4B-LoRA) | Pointer | 轻量 Gemma 版本 |
| [<img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;JevAny-Qwen3.5-4B](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-LoRA) | Pointer | 轻量、支持灵活选项数 |
| [<img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;JevAny-Qwen3.5-4B-Direct-Token](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-Direct-Token-LoRA) | Direct-token | 当前 4B JevBench 最优版本 |
| [<img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;JevAny-Qwen3.8-27B](https://huggingface.co/SimpleJev/JevAny-Qwen3.8-27B-LoRA) | Pointer | 默认模型；当前发布准确率最高 |
| [<img src="docs/model-logos/jevany-muse.svg" width="24" height="24" align="middle" alt="">&nbsp;JevAny-Muse-Glimmer-30B](https://huggingface.co/SimpleJev/JevAny-Muse-Glimmer-30B-LoRA) | Pointer | Muse Glimmer 版本 |

这些 LoRA adapter 采用 SFT 训练，训练数据包含 1,772,725 条文本记录和 2,180,242 个有标签决策，
配置见[训练算力与实验说明](reports/JevAny_Tech_Report.pdf)。
全参数 SFT 与进一步的后训练改进仍在计划中。

加载时还需要对应基座，并适用基座模型的许可证和访问条款。BF16 基座权重大约
需要参数量两倍的字节数，另需运行时显存。详见[硬件与加载说明](docs/DEPLOYMENT.md#checkpoints-and-hardware)。

Pointer 和 direct-token 模型使用相同 API。Pointer 在上下文允许的范围内支持最多
4,096 个选项，direct-token 最多支持 255 个。
训练与准确率的取舍见[输出方式说明](docs/TRAINING.md#pointer-and-direct-token-readouts)。

## 📊 3. 基准测试结果 <a name="评测"></a>

JevAny-Qwen3.8-27B 在两项评测中准确率最高，NLL 和 Brier 也最低。
4B 版本中，direct-token 的 JevBench 准确率最高，Pointer 的 Transfer 准确率最高。

[![JevAny checkpoint 与基线模型按 Transfer 和 JevBench 的平均准确率排序](docs/evaluation-overview.svg)](docs/evaluation-overview.svg)

<div align="center">

| 模型 | Transfer ↑ | JevBench ↑ | NLL ↓ | Brier ↓ | ECE ↓ |
|:---|---:|---:|---:|---:|---:|
| [<img src="docs/model-logos/kev.svg" width="24" height="24" align="middle" alt="">&nbsp;Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) | 74.19% | 75.32% | 0.858 | 0.380 | 0.125 |
| [<img src="docs/model-logos/kev.svg" width="24" height="24" align="middle" alt="">&nbsp;Kev-27B](https://huggingface.co/jaredpalmer/kev-27b) | 82.31% | 85.28% | 0.533 | 0.265 | 0.050 |
| [<img src="docs/model-logos/typesafe.png" width="24" height="24" align="middle" alt="">&nbsp;Jev 1.13.0](https://docs.typesafe.ai/models) | 85.37% | 86.58% | 0.644 | 0.212 | 0.033 |
| [<img src="docs/model-logos/laya.svg" width="24" height="24" align="middle" alt="">&nbsp;Laya](https://huggingface.co/convaiinnovations/laya) | 52.29% | 58.01% | 1.264 | 0.615 | 0.127 |
| **JevAny Releases** |  |  |  |  |  |
| [<img src="docs/model-logos/jevany-gemma.svg" width="24" height="24" align="middle" alt="">&nbsp;JevAny-Gemma-4B](https://huggingface.co/SimpleJev/JevAny-Gemma-4B-LoRA) | 70.84% | 77.49% | 0.706 | 0.369 | 0.056 |
| [<img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;JevAny-Qwen3.5-4B](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-LoRA) | 78.68% | 80.09% | 0.587 | 0.297 | 0.035 |
| [<img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;JevAny-Qwen3.5-4B-Direct-Token](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-Direct-Token-LoRA) | 78.20% | 80.95% | 0.564 | 0.291 | **0.029** |
| [<img src="docs/model-logos/jevany-muse.svg" width="24" height="24" align="middle" alt="">&nbsp;JevAny-Muse-Glimmer-30B](https://huggingface.co/SimpleJev/JevAny-Muse-Glimmer-30B-LoRA) | 83.46% | 87.45% | 0.464 | 0.229 | 0.032 |
| [<img src="docs/model-logos/jevany-qwen.svg" width="24" height="24" align="middle" alt="">&nbsp;**JevAny-Qwen3.8-27B**](https://huggingface.co/SimpleJev/JevAny-Qwen3.8-27B-LoRA) | **86.04%** | **90.04%** | **0.388** | **0.195** | **0.026** |

NLL、Brier 和 ECE 均在 Transfer 上计算。

</div>

[完整结果与评测协议](docs/EVALUATION.md#model-family-v2) ·
[机器可读结果](results/model-family-v2.json) ·
[方法与消融实验报告](reports/JevAny_Tech_Report.pdf)

## 🕹️ 4. 示例与测试环境 <a name="示例与测试环境"></a>

交互演示包含以下三个环境。动图保留历史模型的动作和选项概率；当前
[JevAny-Qwen3.8-27B](https://huggingface.co/SimpleJev/JevAny-Qwen3.8-27B-LoRA)
可按 playground 指南中的命令运行。

### 🤖 4.1 [机械臂插孔](examples/README.md#robot-peg-insertion) <a name="机械臂插孔"></a>

控制 Franka 夹爪抓取、对准并插入工件，由 PyBullet 接触物理验证结果。

[![机械臂浏览器回放：Franka 插孔动作、模型原始选项概率和物理成功检查](docs/demos/playground-arm.gif)](examples/README.md#robot-peg-insertion)

### 🔫 4.2 [Doom 走廊 · 3D](examples/README.md#doom-corridor-3d) <a name="doom-走廊-3d"></a>

击败最后一个房间中左右两侧的敌人，再向前移动。使用 ViZDoom 和随包提供的 Freedoom 资源。

[![Doom checkpoint 回放：击杀左右两名敌人后继续前进](docs/demos/playground-doom.gif)](examples/README.md#doom-corridor-3d)

### ⛏️ 4.3 [Crafter 生存建造 · 2D](examples/README.md#crafter-survival-2d) <a name="crafter-生存建造-2d"></a>

采集木材、制作工具、开采石头，同时管理生命值和物资。

[![Crafter 浏览器回放：资源采集、制作工具和四项目标的完成进度](docs/demos/playground-crafter.gif)](examples/README.md#crafter-survival-2d)

### 🎮 4.4 打开交互演示 <a name="打开交互演示"></a>

按[本地体验](#本地体验)启动模型后，打开交互演示：

```bash
jevany demo --base-url http://127.0.0.1:8008 --text-only
```

打开 `http://127.0.0.1:8090`，点击 **Test and connect**，尝试自己的决策任务。
要让模型操作游戏，安装可选引擎并重新启动演示：

```bash
python -m pip install -e '.[demo]'
jevany demo --base-url http://127.0.0.1:8008 --text-only
```

在浏览器中选择 **Run model**，再点击 **One decision** 单步运行，或
**Run automatically** 连续运行。选择 **Play yourself** 可以自己操作。
实时控制向模型发送文本状态；机械臂控制使用 `.[robotics]` 依赖。

观看内置录制内容时，运行 `jevany demo` 并选择 **Replay**，只需 CPU，无需模型权重。
平台要求与环境接口见[演示指南](examples/README.md)，结合 LLM 规划器使用
JevAny 决策可参考[集成文档](docs/INTEGRATIONS.md)。

## 🧩 5. 支持的模型系列 <a name="支持的模型系列"></a>

[模型 ID、支持的输入与运行要求](docs/TRAINING.md#backbone-support)。

[![支持的 26 个模型，涵盖 Qwen、Gemma、Muse、Mistral、GLM、Nemotron 和 Llama](docs/supported-model-families.svg)](docs/TRAINING.md#backbone-support)

## 📚 6. 文档与贡献 <a name="文档与贡献"></a>

[训练](docs/TRAINING.md) · [部署](docs/DEPLOYMENT.md) · [API](docs/API.md) · [数据](docs/DATA.md) · [评测](docs/EVALUATION.md) · [Agent harness 协议](docs/experiments/AGENT_HARNESS_FRONTIER_PROTOCOL.md) · [贡献指南](CONTRIBUTING.md)

欢迎贡献模型适配、评测或应用示例，开发步骤见[贡献指南](CONTRIBUTING.md)。
[合并版技术报告](reports/JevAny_Tech_Report_with_Agent_Harness.pdf)介绍了模型设计、多模态路径、
agent-harness 实验、负面结果与开放问题；[原始发布报告](reports/JevAny_Tech_Report.pdf)保持不变。

代码和入门数据采用 Apache-2.0。部分组件改编自 [Kev](https://github.com/jaredpalmer/kev)，
归属说明见 [NOTICE](NOTICE) 和 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)。
基座模型与上游数据集保留各自条款。
