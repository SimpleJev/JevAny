---
library_name: peft
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
tags:
  - decision-model
  - lora
  - calibration
  - agents
---

# JevAny-27B-SFT v0.2

[JevAny-Qwen3.8-27B](https://huggingface.co/SimpleJev/JevAny-Qwen3.8-27B-LoRA) is the current recommended general JevAny checkpoint. It uses a rank-8 LoRA adapter and a residual pointer head with a readout dimension of 256 on `Qwen/Qwen3.8-27B`. It maps shared state and typed questions to option probabilities without decoding answer tokens.

## Intended Use

Use this checkpoint for bounded classification, routing, ranking, tool choice, agent actions, and confidence-aware decisions. It accepts choice, noul, and ordinal score questions through the JevAny API.

## Training

The adapter and residual pointer head were trained with supervised fine-tuning on **1,772,725 text records** containing **2,180,242 labelled decisions**. The corpus spans preference, agent and tool decisions, reasoning, classification, and safety. The base weights remain frozen. See the [current training corpus](DATA.md#current-release-scale) for the release scope and data disclosure.

## Evaluation

| Evaluation | Result |
|---|---:|
| Transfer accuracy (1,046 clean, knowable decisions) | 86.04% |
| JevBench public-development accuracy (231 items) | 90.04% |

Transfer uses Kev's frozen `transfer-v9` suite and also informed model development. JevBench reports accuracy on all 231 public development items; it is not a sealed Benchmark Heaven score. Full measurements and suite identifiers are in the [release results](../results/model-family-v2.json), with protocols in the [evaluation guide](EVALUATION.md#model-family-v2).

## Usage

From the repository root, install the serving dependencies and load the published adapter:

```bash
python -m pip install -e '.[serve,multimodal]'
jevany serve --checkpoint SimpleJev/JevAny-Qwen3.8-27B-LoRA --device cuda --dtype bf16
```

## Limits

The released path accepts text, JSON-renderable state, native images, and native video. Multimodal requests currently contain one isolated question and use a bounded visual token budget. HTTP media is operator opt-in through a controlled local root; network media URLs are rejected. Requests must fit the checkpoint's context and configured token limits; see the [deployment guide](DEPLOYMENT.md#native-media-and-limits). Calibration can shift under new data, so validate thresholds on a deployment-specific calibration set. The adapter requires the separately distributed Qwen base weights and JevAny runtime.
