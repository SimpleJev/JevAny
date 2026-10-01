# A small decision-model experiment

This example trains, evaluates and reloads a model using only the public
repository and a public backbone. The starter contains original synthetic
support tickets with separate training and development files. It is useful for
trying the research workflow; replace it with your task's data for meaningful
accuracy comparisons.

Run these commands from the repository root with Python 3.12 or newer:

```bash
python -m pip install -e '.[train,serve]'
jevany data init --out data/research
jevany data validate data/research/train.jsonl
jevany train --config recipes/sft.toml \
  --data data/research/train.jsonl \
  --eval-data data/research/development.jsonl \
  --device cpu --dtype fp32 --weights-dtype fp32 \
  --accum 1 --max-steps 12 \
  --eval-every-steps 4 --checkpoint-every-steps 4 \
  --out runs/research
jevany eval --run runs/research --data data/research/development.jsonl \
  --device cpu --out runs/research-evaluation
jevany decide examples/request.json --checkpoint runs/research --device cpu
```

The recipe uses Qwen3.5-0.8B. The first run downloads its weights; subsequent runs
can use the Hugging Face cache. On a CUDA GPU with enough memory, omit the three
CPU overrides to use the recipe's BF16 settings.

Inspect `data_admission.json` before comparing experiments: it lists any records
excluded by token limits. `training_eval/history.jsonl` records intermediate
held-out scores, and `training_metrics.json` records the examples and optimizer
steps actually processed. The standalone evaluation writes per-question
predictions, metrics and coverage to `runs/research-evaluation`.

To reproduce the rest of this experiment from its fourth optimizer step:

```bash
jevany train --resume runs/research/checkpoints/step-000004 \
  --out runs/research-resumed
```

Resume restores the saved recipe, optimizer, schedule, data order and random
states. Use the same device type and software environment, and retain the
original input files. See [resume semantics](TRAINING.md#resume-an-interrupted-experiment)
for the settings that can change.

## Compare the training paradigms

The default run uses the pointer head with supervised labels. For a direct-token
baseline, repeat the training command with `--decision-mode lm_token` and a new
output directory. This mode uses the base model's original vocabulary head and
single-token candidate labels.

For an RLCR continuation, start from the pointer checkpoint:

```bash
jevany train --config recipes/sft.toml \
  --data data/research/train.jsonl --eval-data data/research/development.jsonl \
  --init-from runs/research --rlcr \
  --device cpu --dtype fp32 --weights-dtype fp32 \
  --accum 1 --max-steps 12 --eval-every-steps 4 \
  --out runs/research-rlcr
```

`--init-from` starts a new optimizer and schedule. It is the entry point for
changing objectives or adapting an existing checkpoint to new data. `--resume`
continues the same experiment. Score each model on the same held-out file and
compare accuracy, NLL, Brier score and coverage, keeping token limits fixed.

## Change a loss or decision head

The trainer's loss functions are ordinary PyTorch functions in
[`train.py`](../jevany/train.py). Start a loss experiment with `question_loss`,
`lm_token_loss` or `rlcr_question_loss`; `batch_loss` selects the objective and
averages question losses within each record.

For example, a label-smoothing ablation can keep the existing soft-target branch
of `question_loss` and change its hard-label return to:

```python
return F.cross_entropy(z[None], y, label_smoothing=0.1)
```

Record that source revision with the experiment and use a new output directory.
Changing a training loss does not change the inference checkpoint format. Resume
assumes that the objective's implementation is unchanged.

For a head ablation, the built-in residual variant already has a recipe setting:

```bash
jevany train --config recipes/sft.toml \
  --data data/research/train.jsonl --eval-data data/research/development.jsonl \
  --device cpu --dtype fp32 --weights-dtype fp32 \
  --head-residual-dim 128 --accum 1 --max-steps 12 --eval-every-steps 4 \
  --out runs/residual-head
```

For a different architecture, the implementation points are:

| Change | Source contract |
|---|---|
| Compute decision logits | `PointerHead` and `DecisionModel._question_readout` in `model.py`; return one logit per option |
| Construct the head | `DecisionModel.__init__` in `model.py`; retain the output interface expected by training and inference |
| Save and reload its configuration | `Meta`, `Checkpoint.load` and `Checkpoint.COMPAT_FIELDS` in `checkpoint.py`, plus the trainer's `Meta` construction |
| Change backbone loading or native media processing | The [BackboneAdapter extension](TRAINING.md#backbone-support) |

Check both hard labels and soft targets for a new loss. For a new head, train a
small model, save it, reload it in a fresh process and compare local and HTTP
answers. Add new architecture fields to warm-start compatibility checks so that
loading mismatched head weights fails explicitly.

## Data provenance

The public starter and dataset builders are available for reproducible workflow
experiments. They do not reconstruct the release's full training mixture. The
[data guide](DATA.md) describes the public inputs and the release-data boundary;
the [evaluation guide](EVALUATION.md) records published results.
