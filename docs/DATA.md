# Data

Training and inference share one JSON structure. Training rows add `label` and may add `target`. Store one object per line in UTF-8 JSONL.

For a dataset you can use immediately, run `jevany data init --out data/starter`.
It copies 24 original synthetic training records and 8 development records from
the installed package, with all three question types and provenance. This starter
is for learning the workflow. Validate it with
`jevany data validate data/starter/train.jsonl`.

For larger data, `jevany data build-sft` and `jevany data build-rlcr` expose the
public-source builders from an installed package. See [TRAINING.md](TRAINING.md)
for a text-only build and recipe commands.

## Import a local multiple-choice export

`jevany data convert` works in the base installation, without downloading a
dataset or loading a model. Each input line contains a question, options and
an answer:

```json
{"question":"Which direction reaches the station?","hint":"The station is south.","options":["north","south"],"answer":"B","group_id":"station-route"}
```

Save this as `questions.jsonl`, then import one explicit split:

```bash
jevany data convert questions.jsonl --out data/my-task \
  --source my-task --revision export-1 --split train
jevany data validate data/my-task/train.jsonl
jevany data check-suite data/my-task
```

The output directory contains `train.jsonl` and `manifest.json`, with source
revision, input checksum, record counts and an output checksum. Convert each
upstream split to a new directory; the command does not shuffle or split data,
skip invalid rows, or overwrite an existing directory. Use the training file
with `jevany train --data` and a separate development file with `--eval-data`.
The trainer performs token-context admission when training starts.

Each conversion directory contains one partition. `check-suite` compares
partitions listed together in one suite manifest; it does not compare separate
conversion directories.

`options` can be a list or its Python-literal string representation.
For named keys, use `choices: {"text": [...], "label": [...]}`.
`answer` or `answerKey` resolves in this order: an exact option key, exact
option text, a letter (`A`, `B`, …), or a zero-based index. Missing answers,
negative indices, duplicate keys and mismatched text/label arrays are rejected.
Omit `label` to generate index keys; an explicitly empty label array is invalid.
`question` may also be an object with a `stem` string.

Optional `hint`, `fact1`, `fact2` and `combinedfact` fields become evidence.
Use the same `group_id` for related rows; conversion preserves it under the
source name, including when converting different upstream splits.
Labels and provenance remain outside the model-facing request.
Conversion errors identify the input file and line.

## Question Types

### Choice

```json
{
  "type": "choice",
  "instructions": "Which team should handle this?",
  "criteria": {
    "billing": "Charges and payment problems",
    "shipping": "Delivery delays"
  },
  "label": "billing"
}
```

The label is one key from `criteria`.

### Noul

```json
{
  "type": "noul",
  "instructions": "Is manual review required?",
  "label": true
}
```

`noul` is a binary probability. Its label is `true` or `false`.

### Score

```json
{
  "type": "score",
  "instructions": "How urgent is this?",
  "criteria": ["low", "normal", "high"],
  "label": 2
}
```

The label is the zero-based level index.

## Soft Targets

Use a soft target when the evidence does not support one certain answer:

```json
{
  "type": "noul",
  "instructions": "Is the parcel late under the promised service level?",
  "label": false,
  "target": {"false": 0.5, "true": 0.5}
}
```

The `label` remains required for evaluation compatibility. Training uses `target` when present. Values are normalized after loading. Targets may omit zero-weight options; unknown keys, negative or nonfinite weights, and zero total mass are rejected, along with labels outside the question's options.

## Native Media

Attach native image or video evidence at the request level. A multimodal request currently contains exactly one isolated question.

```json
{
  "state": {"study": "Inspect the diagram before answering."},
  "media": [{"type": "image", "uri": "cases/diagram.png"}],
  "questions": {
    "answer": {
      "type": "choice",
      "instructions": "Which component is connected to the battery?",
      "criteria": {"a": "Motor", "b": "Lamp"},
      "label": "b"
    }
  }
}
```

Training and frozen suites resolve relative paths against the JSONL directory. The HTTP server requires `JEVANY_MEDIA_ROOT` to enable media and accepts only local files inside that directory. Network URLs are rejected. File bytes, total request bytes, pixels, and declared video frames have configurable caps; videos without a declared frame count are rejected. Use a dedicated upload directory as the media root.

## Full Record

See [`examples/train.jsonl`](../examples/train.jsonl). `state` and `instructions` may be strings, objects, arrays, numbers, booleans, or null. Object field names are preserved as text labels. Every question should be answerable from the state and instructions alone.

When creating data:

- keep option keys stable and descriptions specific;
- include cases where information is genuinely missing;
- distinguish ambiguity from label noise;
- keep train and evaluation sources separate, then check normalized text hashes;
- preserve source, generator, prompt, verifier, and license metadata outside the model-facing fields;
- audit generated labels and counterfactual pairs before training.

## Current Release Scale

The five LoRA SFT releases listed in the README's Pretrained Models table
belong to `model-family-v2`. They were trained with supervised fine-tuning on
**1,772,725 text records** containing **2,180,242 labelled decisions**. See the
[release metadata](../results/model-family-v2.json) for the recorded counts and
model list. At a high level, the corpus covers:

- preference and ranking decisions;
- agent, tool-use, and action selection;
- general and domain reasoning;
- classification and policy decisions; and
- safety-sensitive choices.

These are the same aggregate counts and task families reported in the project
README and the [technical report](../reports/JevAny_Tech_Report.pdf).

JevAny also supports native image and video records through the same request
schema. Media are resolved and passed to each compatible backbone's native
processor; support is declared by the loaded checkpoint rather than inferred
from a filename or prompt.

The bundled `build-sft` and `build-rlcr` commands remain reference builders for
public experiments and custom training. They do not reconstruct the current
release corpus. Review upstream licenses before downloading, training on, or
redistributing any converted data.

## Evaluation Separation

For a frozen suite with a `manifest.json`, run `jevany data check-suite PATH`.
It checks declared partitions' checksums and counts, labelled question validity,
source restrictions, and overlap in normalized decision text or group IDs across
partitions. It reports the partitions checked and any skipped partition.
The locked test partition is not read unless you pass `--allow-test`, so
train/test overlap is checked only with that flag.
A suite containing only the locked test partition requires that flag.

Keep training, calibration, and evaluation records separate. Fit calibration
parameters only on the calibration partition, preserve group identifiers for
related questions, and check normalized text hashes before evaluation. Current
release metrics and suite hashes are recorded in
[`results/model-family-v2.json`](../results/model-family-v2.json); protocol notes
are in [EVALUATION.md](EVALUATION.md).
