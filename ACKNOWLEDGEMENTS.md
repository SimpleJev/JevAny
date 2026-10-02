# Acknowledgements

JevAny includes modified infrastructure code from [Kev](https://github.com/jaredpalmer/kev), created by Jared Palmer and released under Apache-2.0. The required attribution is also recorded in `NOTICE` and in derived source files.

JevAny develops a separate research direction around calibration-aware reinforcement learning for decision models. Tianxin Wei created the current training system, RLCR implementation, checkpoints, evaluation, release tooling, and product roadmap.

The architecture was informed by the public analysis in [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked) and the public System One API.

RLCR follows the reward proposed in *Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty* by Damani et al. JevAny adapts it to a prefill-only pointer model.

Qwen3.8-27B is produced by the Qwen team. The adapters in this project require its separately distributed base weights.

The Phi-4 vision conversion helper adapts configuration and weight-name mappings from Hugging Face Transformers under Apache-2.0. It merges Microsoft's pretrained vision LoRA before JevAny training.

The native Phi-4 Reasoning Vision adapter follows Microsoft's [published model layout and image preprocessing](https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B/tree/c3e4fac79ddace21976ced56fbf1564b8bd8c89f), released under MIT. Its weights are downloaded separately from Microsoft.

The experimental training-free option-letter readout adapts prompt and probability-aggregation semantics from [Cygnet](https://github.com/blockbrain-ai/cygnet-recipe) at commit `3cf591c692dec649f7c134449814610307c7bb3a`, Copyright 2026 Nood Co and contributors, under MIT. Cygnet credits the one-token option-letter readout method to [NInfer](https://github.com/igorls/ninfer), released under Apache-2.0. JevAny implements its own model integration and does not incorporate NInfer source code. The full Cygnet notice is in `NOTICE`.

The README teaser uses Lucide icons. The [source records](docs/icons/sources.json) and [license notices](docs/icons/LICENSE) accompany the editable SVG.

The supported-model cards use logos from [Lobe Icons](https://github.com/lobehub/lobe-icons) under MIT. Their [source records](docs/model-logos/sources.json) and [license](docs/model-logos/LICENSE) accompany the SVG. Model and publisher marks belong to their respective owners.
