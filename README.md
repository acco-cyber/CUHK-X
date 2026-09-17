# Structure-Aware Probabilistic Video-QA Decoder

> **Structure-aware VQA decoder with duration-aware manner inference**
>
> A deterministic, CPU-only solution that treats the benchmark as a structured inference problem rather than a generic vision-language generation task. The system reconstructs scene scripts from the labelled training set, decodes questions jointly, and uses exact IR-video duration metadata to resolve a small set of ambiguous manner assignments.

<div align="center">

**Primary selected submission:** `56255655`  ·  **Public LB:** `0.99707`  ·  **Private LB:** `0.98529`

**Model class:** structured probabilistic decoder + Ridge regression  ·  **Neural checkpoint:** none  ·  **GPU:** none

</div>

---

## Table of Contents

- [What this repository contains](#what-this-repository-contains)
- [Architecture](#architecture)
- [Method at a glance](#method-at-a-glance)
- [Model and learned state](#model-and-learned-state)
- [Training procedure](#training-procedure)
- [Inference procedure](#inference-procedure)
- [Stage-by-stage implementation](#stage-by-stage-implementation)
- [Results and validation](#results-and-validation)
- [Reproducing the selected submissions](#reproducing-the-selected-submissions)
- [Repository map](#repository-map)
- [Runtime and hardware](#runtime-and-hardware)
- [Artifacts and checksums](#artifacts-and-checksums)
- [Provenance and disclosure](#provenance-and-disclosure)
- [Determinism and reproducibility](#determinism-and-reproducibility)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Final verification checklist](#final-verification-checklist)

---

## What this repository contains

This repository is a post-deadline consolidation of the research code that produced the selected CUHK-X Large Model Track submissions. It is intentionally lightweight: the executable solution uses **Python standard library only**, requires **no CUDA**, has **no third-party inference dependency**, and does not load a neural model.

The core idea is to exploit regularities in the benchmark construction:

- HAU clips form repeated **scene scripts**.
- Consecutive recordings reveal the **repeat index**.
- Training questions expose the **action set**, **action precedence**, and **manner/adverb structure** of a scene.
- Several test participants **re-perform training scripts**, creating useful training/test twins.
- For scenes without a matched twin, **IR clip duration** provides an additional signal for identifying manner order.

The final system therefore performs structured decoding over all questions belonging to a scene, rather than treating each multiple-choice question independently.

<details open>
<summary><strong>Result snapshot</strong></summary>

| Item | Value |
|---|---|
| Track | `large` |
| Team | `Nabid Nur` (`koushikrudra`, `nabidnur`) |
| Primary selected submission | `56255655` |
| Primary file | `text_decoder_v6.csv` |
| Public score | `0.99707` |
| Private score | `0.98529` |
| Inference architecture | Structured probabilistic decoder |
| Statistical learner | Ridge regression on within-scene log duration |
| Neural network | None |
| Pretrained model / tokenizer | None |
| GPU | 0 |
| Python dependencies | Standard library only |
| Inference-time API / LLM | None |

The companion notebook reports `0.99707 (341/342)` on the public leaderboard for v6 and documents the v3 → v4 → v6 progression. The README deliberately keeps the provenance and disclosed manual decisions visible rather than presenting the score as a purely automated neural-model result.

</details>

---

## Architecture

<details open>
<summary><strong>Open full architecture diagram</strong></summary>

<p align="center">
  <img src="docs/architecture.svg" alt="CUHK-X Large Model Track v6 architecture" width="1100">
</p>

The diagram is organized as a research pipeline:

**Training evidence → knowledge store → six-stage decoder/learner → posterior fusion → submission**

The duration branch is deliberately separated from the structural decoder. The system only uses duration metadata where text structure leaves an ambiguity, which is why the final v6 change set is small relative to the full submission.

</details>

<details>
<summary><strong>Architecture details</strong></summary>

### Inputs

**Training evidence**

- `training_qa.csv`: labelled HAU + HARn questions.
- `HAU.zip`: official training media archive used only to recover IR clip duration metadata.
- Training scenes define the statistical knowledge used by stages 1–5.

**Test evidence**

- `test_qa.csv`: 682 label-free questions.
- 208 test clips in the official test archive.
- Only the small `IR.mp4` member is inspected for duration; video pixels are never decoded.

### Learned knowledge store

The structured decoder builds a deterministic state containing, among other quantities:

- Scene scripts `S = (A, E, P, n)`.
- `A`: scene action set.
- `E[z]`: manner/adverb associated with repeat `z`.
- `P`: precedence closure between actions.
- `P(adverb | repeat)` counts with manner-class smoothing.
- HARn label/object statistics.
- Action support and co-occurrence counts.

### Six execution stages

| Stage | Module | Role |
|---|---|---|
| **1** | `text_decoder.py` | Scene grouping, twin-script matching, joint HAU decoding, HARn monotone DP |
| **2** | `slot_rule.py` | Recover missing temporal-order pairs using substitute-action slots |
| **3** | `scene_joint.py` | Jointly enforce single / combination / multi scene consistency |
| **4** | `singleton.py` | Marginalize repeat position for one-clip no-twin scenes |
| **5** | `duration.py` | Fit Ridge duration model and decode manner assignments with duration likelihood |
| **6** | `pipeline.py` | Apply the explicitly disclosed row adjustments for the exact original test set; not a model stage |

### Output

`prediction` is generated in the exact order of `test_qa.csv` and written as:

```text
qa_id,prediction
```

</details>

---

## Method at a glance

<details open>
<summary><strong>1. Scene grouping</strong></summary>

HAU clips are grouped in recording order. Consecutive clips whose emotion-option sets share the scene's manner candidates are treated as repeats of the same script. The repeat position is therefore an observable structural feature, not a learned visual representation.

For two-clip scenes the frozen solution uses:

```text
z = 2, 3
```

This choice was established during development and is documented explicitly in the disclosure section.

</details>

<details>
<summary><strong>2. Twin-script matching</strong></summary>

Each test scene is compared with the training scene scripts using structural compatibility:

- action-set overlap,
- emotion-option overlap,
- sequence compatibility,
- scene consistency,
- action-precedence information.

When a twin is found, the corresponding training script supplies high-value latent structure: actions, manner by repeat, and temporal order.

The training knowledge base contains **272 HAU scene scripts**.

</details>

<details>
<summary><strong>3. Joint constraint decoding</strong></summary>

Questions from the same scene are solved jointly because the dataset contains strong consistency rules.

The observed training regularities used by the decoder include:

- a single-question distractor is almost never a scene action (`1 / 2,427` observed exception);
- a false multi-select option is almost never a scene action (`1 / 1,729` observed exception);
- exactly one combination option consists only of scene actions in `789 / 790` observed cases.

These rules are used as constraints and compatibility scores rather than as independent question-level classifiers.

</details>

<details>
<summary><strong>4. Sequence-order inference</strong></summary>

For HAU temporal-order questions, training scenes induce pairwise precedence relations such as:

```text
A → B
B → C
A → C
```

The decoder computes a transitive closure over these relations. For missing pairs, the slot rule identifies substitute actions that occupy an equivalent temporal slot in other training scenes and transfers the supported precedence.

For HARn, the solution uses a **softly monotone dynamic programme**. Label candidates are scored using Naive-Bayes-like option statistics, and the optimal label path is constrained by the known label-sorted ID structure.

</details>

<details>
<summary><strong>5. Duration-aware manner inference</strong></summary>

For no-twin scenes, the text decoder can still leave multiple plausible mappings between the three manner adverbs and repeat positions. v6 adds a statistical duration model that uses only metadata extracted from the IR videos.

The central empirical observation is simple:

```text
faster manner  →  shorter recording
later repeat   →  shorter recording on average
```

The notebook reports **89 / 93** directionally consistent clip-duration comparisons and **37 / 39** scenes where the fastest manner was also the shortest clip among the evaluated twin scenes.

</details>

---

## Model and learned state

### What this is — and what it is not

This is **not** a Transformer, CNN, ViT, CLIP model, LLM, or video encoder. There is no pretrained checkpoint, no tokenizer, and no neural parameter tensor.

The final system is better characterized as:

> **Structured probabilistic inference + deterministic rules + a small Ridge regression learner**

The shipped `artifacts/model_state.json` is therefore a serialized **knowledge/state checkpoint**, not a neural network weight file.

### Duration model

The duration learner is defined as:

\[
\log d_{s,z}
= \mu_s
+ c(\text{manner})
+ \delta(\text{adverb})
+ f(z)
+ \epsilon
\]

where:

- `d` is the IR clip duration in seconds;
- `μ_s` is a scene-specific baseline;
- `c(manner)` is a class effect for `slow`, `careful`, or `fast`;
- `δ(adverb)` is the adverb-specific deviation;
- `f(z)` is the repeat-position effect for `z ∈ {1,2,3}`;
- `ε` is the residual term.

The implementation centers both features and targets within each scene so that the unknown absolute scene-length baseline cancels during learning.

### Ridge objective

The duration model applies a Ridge penalty to the adverb-specific terms:

\[
\hat\beta = \arg\min_{\beta}
\left(\|y-X\beta\|_2^2 + \lambda\|\beta_{adverb}\|_2^2\right)
\]

with:

```text
λ = 4.0
```

The class and repeat effects receive only a tiny numerical stabilizer in the implementation.

<details>
<summary><strong>Final fitted duration coefficients</strong></summary>

The notebook's final fit on all eligible training observations reports:

| Learned term | Coefficient in log-duration space |
|---|---:|
| `slow` | `+0.099` |
| `careful` | `+0.127` |
| `fast` | `-0.225` |
| repeat `z=1` | `+0.122` |
| repeat `z=2` | `-0.020` |
| repeat `z=3` | `-0.102` |
| residual `σ` | `0.120` |
| Ridge penalty `λ` | `4.0` |

The final learner was fit on **272 scenes / 809 eligible clips**. The duration extraction pass produced **814 training IR durations** in total; only clips participating in duration-bearing emotion scenes enter the fitted learner.

</details>

### Duration-aware decoding

For an admissible hypothesis `h` that assigns manner adverbs to repeat positions, the decoder evaluates:

\[
\text{Score}(h)
=
\sum_z \log P(a_z\mid z)
+
 w\,\mathcal{L}_{Gaussian}(\text{centered log-duration}\mid h)
\]

with the final frozen value:

```text
w = 1.0
```

For two-repeat scenes, the solution evaluates the observed repeats as `z = 2,3` and treats the remaining manner adverb as the dropped `z = 1` assignment when constructing the hypothesis lattice.

### Why the duration signal is effective

The duration branch is not a generic video model. It only asks the MP4 header for the exact movie duration:

```text
ZIP
  └── IR.mp4
        └── moov
              └── mvhd
                    └── duration (seconds)
```

No video frame is decoded, and the visual stream is not passed through a neural encoder.

---

## Training procedure

<details open>
<summary><strong>Open the complete training pipeline</strong></summary>

### Step 1 — Build labelled scene state

`src/train.py` reads:

```text
Training/training_qa.csv
Training/data/HAU.zip
```

and reconstructs the HAU clips and scene scripts.

The state contains:

- action sets,
- action support counts,
- pairwise and transitive precedence relations,
- manner/adverb answer frequencies,
- adverb-option frequencies,
- HARn label/object statistics.

### Step 2 — Recover duration metadata

The media reader does not extract the entire archive. It:

1. range-reads the ZIP end-of-central-directory;
2. reads the central directory;
3. locates each `IR.mp4` member;
4. fetches only the compressed member bytes needed for that clip;
5. inflates the member if required;
6. parses the MP4 `moov/mvhd` box;
7. records duration in seconds.

The training pass in the companion notebook read **814 IR members** from a **3.67 GB** archive using **1,631 range requests**.

### Step 3 — Construct duration-learning observations

A training observation is created only when the HAU scene has:

- an emotion question,
- a readable duration,
- at least two repeats available for the centered-duration fit.

This produced **272 duration-bearing scenes and 809 fitted clips**.

### Step 4 — Fit the Ridge model

For each scene, duration and feature vectors are centered around the scene mean. This removes the absolute scene baseline `μ_s` and leaves the relative duration pattern needed for manner/repeat inference.

The solver is implemented directly in Python with deterministic linear algebra. There is no scikit-learn dependency.

### Step 5 — Select the duration weight

The duration fusion weight was evaluated on held-out training groups using leave-one-group-out validation. The tested values were:

```text
w ∈ {0, 0.5, 1, 2}
```

The frozen solution uses `w = 1.0` because it reached the best held-out performance among the tested settings and maintained the strongest blind-twin result at that setting.

### Step 6 — Serialize model state

The deterministic result is written to:

```text
artifacts/model_state.json
```

This file is a stateful statistical checkpoint, not a neural checkpoint.

</details>

---

## Inference procedure

<details open>
<summary><strong>Open the full inference flow</strong></summary>

```text
                    test_qa.csv
                         │
                         ▼
              ┌────────────────────┐
              │  Stage 1           │
              │  Scene grouping +  │
              │  script matching   │
              └─────────┬──────────┘
                        │
             ┌──────────┴──────────┐
             │                     │
          twin scene           no-twin scene
             │                     │
             ▼                     ▼
      known script state      text-only constraints
             │                     │
             └──────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ Stage 2            │
              │ Sequence slot rule │
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ Stage 3            │
              │ Scene joint rules  │
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ Stage 4            │
              │ Singleton prior    │
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ Stage 5            │
              │ Duration + Ridge   │
              └─────────┬──────────┘
                        ▼
                  final model output
                        │
                        ▼
              ┌────────────────────┐
              │ Stage 6            │
              │ disclosed rows     │
              └─────────┬──────────┘
                        ▼
                  submission.csv
```

The model computes all stage 1–5 answers deterministically. Stage 6 exists only to make the post-deadline research package reproduce the exact selected CSVs; it is explicitly **not** presented as a learned model stage.

</details>

---

## Stage-by-stage implementation

<details open>
<summary><strong>Stage 1 — Structural text decoder</strong></summary>

`src/cuhkx_vqa/text_decoder.py`

The decoder learns the benchmark's structural grammar from the training set and uses it to solve clips jointly.

For HAU:

- group consecutive clips into scenes;
- infer repeat position from recording order;
- match candidate test scenes against training scripts;
- use scene action sets and emotion mappings from matched twins;
- solve single / multi / combination questions jointly.

For HARn:

- build label/object statistics;
- score candidate labels for each clip;
- solve the entire sequence with a softly monotone dynamic programme.

</details>

<details>
<summary><strong>Stage 2 — Sequence slot rule</strong></summary>

`src/cuhkx_vqa/slot_rule.py`

When a sequence question contains action pairs that the matched twin script does not explicitly cover, the solver searches the training corpus for substitute actions that:

- do not co-occur in the same sequence question,
- appear in multiple corresponding scenes,
- occupy the same structural slot against other known actions.

The inferred pairs are added through transitive closure, and the highest-scoring consistent permutation is retained.

Held-out validation reported **42 / 43** correctly recovered missing-pair cases.

</details>

<details>
<summary><strong>Stage 3 — Scene-joint consistency</strong></summary>

`src/cuhkx_vqa/scene_joint.py`

For no-twin scenes, the decoder jointly chooses answers so that all questions agree on a single scene action set.

The key rules are:

- selected single answers become candidate scene actions;
- combination answers must respect the inferred action set;
- multi answers are derived from the resulting action support;
- distractor options are penalized or excluded when sibling questions already identify the true actions.

Held-out validation reported **51 fixed / 0 broken** on the evaluated three-repeat cases and **52 fixed / 3 broken** on the evaluated two-repeat cases.

</details>

<details>
<summary><strong>Stage 4 — Singleton inference</strong></summary>

`src/cuhkx_vqa/singleton.py`

A one-clip no-twin scene does not reveal which of repeats 2 or 3 is being observed. The decoder therefore marginalizes over the two admissible positions using smoothed `P(repeat | adverb)` statistics and the manner lexicon.

Held-out validation reported approximately **0.497 vs 0.405** for the evaluated comparison.

</details>

<details>
<summary><strong>Stage 5 — Duration model</strong></summary>

`src/cuhkx_vqa/duration.py`

This stage is the only statistical learner in the final package.

Input features:

```text
manner class(adverb)
adverb identity
repeat index z
```

Target:

```text
centered log(duration)
```

Learner:

```text
Ridge regression
λ = 4.0
```

Decoder:

```text
repeat prior + Gaussian duration likelihood
weight = 1.0
```

Leave-one-group-out results:

| Setting | `w=0` | `w=0.5` | `w=1` | `w=2` |
|---|---:|---:|---:|---:|
| 3-repeat | `0.819` | `0.912` | `0.922` | `0.911` |
| z=2,3 pair | `0.760` | `0.905` | `0.913` | `0.921` |

Blind twin-scene validation also improved from `0.796` at `w=0` to `0.959` at `w=1` and `w=2`.

</details>

<details>
<summary><strong>Stage 6 — Disclosed adjustments</strong></summary>

`src/cuhkx_vqa/pipeline.py`

This stage is a reproducibility layer. It applies the explicitly documented adjustments only when the input matches the original competition test set and the duration stage completes for every eligible scene.

It should not be interpreted as a learned model component.

</details>

---

## Results and validation

### Submission progression

| Version | Main change | Public LB |
|---|---|---:|
| v3 | structural text decoder + sequence slot rule | `0.98245` |
| v4 | scene-consistency fixes + singleton rule | `0.98245` |
| **v6** | **duration-aware emotion decoding for no-twin scenes** | **`0.99707`** |

The companion notebook verifies each intermediate submission against the expected MD5 and reports that v6 differs from v4 in **12 rows** at the model-output level before the separately disclosed rows are considered.

### Held-out validation summary

| Component | Evaluation | Result |
|---|---|---:|
| Scene / twin decoding | 3-repeat held-out twins | `0.971` |
| Scene / twin decoding | 2-repeat held-out cases | `0.87–0.99` depending on missing repeat |
| No-twin structural decoding | 3-repeat held-out scenes | `0.863` |
| HARn decoder | held-out users | `0.994` |
| Sequence slot rule | missing-pair validation | `42 / 43` |
| Singleton rule | held-out comparison | `0.497 vs 0.405` |
| Duration model | 3-repeat LOGO | `0.819 → 0.922` |
| Duration model | 2-repeat LOGO | `0.760 → 0.913` |
| Duration model | blind test twins | `0.796 → 0.959` |

The benchmark-specific construction is doing substantial work here. These numbers should not be interpreted as evidence that the same decoder will obtain comparable accuracy on an unrelated VQA dataset.

---

## Reproducing the selected submissions

### Prerequisites

- Python 3.
- Bash for `inference.sh`.
- Official CUHK-X Large Model Track test data.
- No GPU.
- No `pip install` step.
- No CUDA.
- No external API.

### Recommended command

```bash
bash inference.sh <data_dir> <output_csv>
```

This targets the primary selected submission `56255655`.

To reproduce the second selected submission:

```bash
bash inference.sh <data_dir> <output_csv> 56261134
```

To generate the **model-only** result without stage-6 disclosed adjustments:

```bash
python3 src/predict.py \
  --data-dir <data_dir> \
  --output-csv <output_csv> \
  --config configs/model_only.yaml
```

### Rebuild the model state

```bash
python3 src/train.py --training-dir <Training_dir>
```

This reads `training_qa.csv` and the HAU training archive. When the duration cache is supplied through the training command's supported duration argument, the shipped `artifacts/model_state.json` can be rebuilt byte-for-byte.

### Expected input layout

`<data_dir>` should contain the official label-free test data. The runner accepts either extracted video directories or ZIP archives below the data directory.

Expected logical inputs:

```text
<data_dir>/
├── test_qa.csv
└── ...
    ├── <clip>/IR/IR.mp4
    └── or a .zip containing the clip directories
```

The exact test runner also supports nested layouts and searches for the shallowest valid `test_qa.csv` within the documented depth limit.

### Output format

```csv
qa_id,prediction
...
```

The file is emitted in the same row order as `test_qa.csv`, with UTF-8 text and Python `csv` module default `CRLF` line endings.

---

## Repository map

The following paths are the core implementation surfaces documented by the package:

| Path | Purpose |
|---|---|
| `src/predict.py` | Inference entry point |
| `src/train.py` | Deterministic model-state builder |
| `src/cuhkx_vqa/pipeline.py` | Stage orchestration and final output assembly |
| `src/cuhkx_vqa/text_decoder.py` | Stage 1 structural decoder |
| `src/cuhkx_vqa/slot_rule.py` | Stage 2 sequence-slot inference |
| `src/cuhkx_vqa/scene_joint.py` | Stage 3 scene-joint constraints |
| `src/cuhkx_vqa/singleton.py` | Stage 4 one-clip repeat prior |
| `src/cuhkx_vqa/duration.py` | Stage 5 Ridge duration learner + decoding |
| `artifacts/model_state.json` | Serialized training-derived statistical state |
| `artifacts/disclosed_adjustments_56255655.json` | Disclosed rows for selected submission #1 |
| `artifacts/disclosed_adjustments_56261134.json` | Disclosed rows for selected submission #2 |
| `configs/final.yaml` | Primary submission configuration |
| `configs/submission_56261134.yaml` | Second selected-submission configuration |
| `configs/model_only.yaml` | Stages 1–5 only |
| `inference.sh` | Reproduction wrapper |
| `tests/test_components.py` | Component and input-layout tests |
| `environment/requirements.txt` | Intentionally empty dependency manifest |
| `environment/Dockerfile` | Optional, unverified Docker recipe |
| `docs/architecture.svg` | Full system architecture diagram |

---

## Runtime and hardware

<details open>
<summary><strong>Environment matrix</strong></summary>

| Resource | Requirement / observed value |
|---|---|
| GPU | **None** |
| GPU count | `0` |
| GPU memory | `0 GB` |
| Minimum host memory | `1 GB` |
| Package disk footprint | about `10 MB`, excluding organizer data |
| Linux | x86_64 tested |
| Windows | Windows 11 tested |
| Python | CPython `3.11.9` on Windows; `3.12.13` on Linux/Kaggle |
| Python dependencies | Standard library only |
| CUDA / compiler | Not required |
| Inference API | None |
| LLM service | None |
| Decoder time | about `1–1.5 s` on tested machines, plus a few seconds for duration extraction |
| Video decoding | None; only MP4 metadata is read |

</details>

<details>
<summary><strong>Optional Docker</strong></summary>

A Dockerfile is included, but it was not built or tested by the team. Building it requires network access to pull `python:3.11.9-slim-bookworm`; runtime does not need network access.

```bash
docker build -f environment/Dockerfile -t nabidnur/cuhkx-large-vqa:v1 .

docker run --rm --network none \
  -v <data_dir>:/data:ro \
  -v <output_dir>:/output \
  nabidnur/cuhkx-large-vqa:v1 \
  bash inference.sh /data /output/submission.csv
```

</details>

---

## Artifacts and checksums

| Artifact | Purpose | Size | SHA-256 / reference |
|---|---|---:|---|
| `artifacts/model_state.json` | Training-derived model state | `151,483 B` | `f90c495999a03d9f721fe45ac3ee46cf7aa5d55dea69df961fc95e4f43de9cf4` |
| `text_decoder_v6.csv` / selected #1 | Primary selected submission | — | `c2792401d8fc4c790d92252e537f362365a690052199e263a370c8713577cb1d` |
| `text_decoder_v7A.csv` / selected #2 | Second selected submission | — | `e9bead9015e68442ab6db3ca2007ee2af5d018f5abf4327c21704074fd66e2f0` |
| `v6` notebook checksum | Exact submitted notebook-side CSV | — | MD5 `18d60c107d2c24a1d0246ac789ecd161` |

The disclosure JSON files contain the rows that were not generated by the pure stage-1-to-5 model path.

<details>
<summary><strong>What is inside model_state.json?</strong></summary>

The state file stores the deterministic statistical knowledge required by stages 1–5, rather than weights for a neural network. Conceptually it contains:

```text
scene scripts
action-support counts
pairwise precedence information
transitive action-order relations
adverb / repeat counts
manner-class mapping statistics
HARn label/object statistics
Naive-Bayes-style option statistics
Ridge duration coefficients
Ridge residual scale
configuration values
```

</details>

---

## Provenance and disclosure

<details open>
<summary><strong>Read before evaluating the score</strong></summary>

### Manual / disclosed answers

The selected submissions are not purely model-generated. The package explicitly discloses the rows that were set or changed by hand during development:

- `test_0358`: temporal-order answer set by hand with an AI assistant.
- `test_0681` and `test_0423`: scene `LM_test_0158-0160`, where the duration model's top two hypotheses were nearly tied (`0.53` vs `0.46`); the team selected the second-ranked hypothesis using evidence from an earlier submission.
- `test_0430` in submission `56261134`: changed using duration / bitrate evidence and a consistency analysis over public-leaderboard results.

Without these disclosed rows, the model output differs from the selected `56255655` CSV by exactly 3 rows and from `56261134` by exactly 4 rows.

### Leaderboard-informed development

Several decisions were informed by public-leaderboard feedback during competition. In particular:

1. The two-repeat mapping was fixed to `z = [2,3]` after controlled probes comparing alternative mappings.
2. The selected hypothesis for scene `LM_test_0158-0160` was chosen using public-score evidence after a near tie in the duration model.
3. `test_0430` in `56261134` was changed using additional external evidence.
4. Six temporal-order answers were retained after a public-score improvement in v3.

These facts are intentionally documented because the goal of this repository is accurate reproduction and scientific transparency, not post-hoc presentation of the selected CSV as a fully autonomous output.

### Development-time AI assistance

Claude (Anthropic) was used as a coding and analysis assistant during development, including multi-agent workflows. The assistance included reading training/test structure, writing and validating stages 2–5, and proposing candidate answers for selected test rows.

No LLM, API, or network service is called by the final inference path.

### Data provenance

No test ground-truth labels, external datasets, or pretrained models were used. Duration metadata comes from the official CUHK-X training/test media archives and is read from organizer-hosted ZIPs using HTTP range requests during data preparation.

</details>

---

## Determinism and reproducibility

The solution is designed to be exactly deterministic.

- No randomized training procedure.
- No stochastic decoding.
- No neural inference kernel.
- Integer and floating-point arithmetic are performed in Python.
- `inference.sh` sets `PYTHONHASHSEED=0`.
- The selected CSVs were reproduced byte-for-byte in the documented validation environment.

<details>
<summary><strong>Reproduction caveats</strong></summary>

The team verified the decoder on two platforms:

- Windows 11, CPython 3.11.9.
- Linux x86_64, glibc 2.35, CPython 3.12.13, private Kaggle CPU session.

The exact official test ZIP could not be downloaded again after the competition because the organizer's Drive mirror download quota was exceeded. The byte-identical reproduction check therefore used a stand-in test archive carrying the original `mvhd` duration headers. The real official archive was also checked on its first gigabyte, where 108 IR videos matched the durations used for the submissions.

Three rows rely on exact or last-bit score ties in the deterministic solver: `test_0343`, `test_0647`, and `test_0509`. Evaluation order currently breaks those ties consistently across the tested platforms.

</details>

---

## Limitations

<details open>
<summary><strong>Benchmark dependence</strong></summary>

The method is intentionally specialized to the CUHK-X Large Model Track construction. Its strongest assumptions are:

1. Some participants re-perform training scene scripts.
2. Test clip IDs preserve recording-order information.
3. Scenes use a repeated three-manner structure.
4. Emotion choices expose a reusable adverb set.
5. Duration differences correlate with manner and repeat position.

On a dataset without those properties, the structural match rate will drop and the decoder falls back toward weaker text priors. The code is still designed to produce a valid CSV, but the competition score should not be generalized beyond the benchmark.

</details>

<details>
<summary><strong>What the method does not claim</strong></summary>

- It does not perform general-purpose video understanding.
- It does not learn semantic representations from raw frames.
- It does not use a pretrained foundation model.
- It does not estimate a universal relationship between speaking/action speed and clip duration.
- It does not constitute evidence that a small Ridge model can replace a vision model on unrelated tasks.

The duration learner is a benchmark-specific auxiliary signal.

</details>

---

## Troubleshooting

<details open>
<summary><strong>Common runtime messages</strong></summary>

### `[stage6] disclosed adjustments NOT applied`

The input `test_qa.csv` differs from the original test set, or stage 5 did not complete for every eligible scene. The system then returns the model-only output for that input.

### `[stage5] WARNING: no readable video durations ...`

The test videos are missing or unreadable. Confirm that either extracted clip directories or the relevant ZIP archive exists below `<data_dir>`.

### `[pipeline] WARNING: ... clip directories have no trailing number`

Scene grouping depends on recording order encoded in the clip directory name. Confirm that the expected `LM_test_NNNN` naming is present.

### Bash / line-ending issues

`inference.sh` is distributed with LF line endings. Converting the script to Windows CRLF before running under Bash can cause shell errors.

</details>

---

## Final verification checklist

<details>
<summary><strong>Package readiness</strong></summary>

- [ ] Replace all remaining `REPLACE_ME` fields (including the contact email in `manifest.yaml`).
- [x] `manifest.yaml` matches the documented submission files.
- [ ] Run the full package from a clean directory through `inference.sh`.
- [ ] Confirm the exact Selected submission ID against Kaggle's **Selected** filter before external reproduction.
- [ ] Confirm `primary_verification_submission_id` with the organizing committee if required by the challenge procedure.
- [x] External data, model, API, licensing, and access requirements are documented.
- [ ] Complete and sign `declarations/CUHK-X_Honor_Declaration.docx` as required by the team.
- [x] No API keys, passwords, personal credentials, official raw challenge data, or unnecessary caches are included.

</details>

---

## Citation / attribution note

This repository documents a competition solution whose main contribution is the **use of benchmark structure as a source of inference constraints**, with **duration metadata as a targeted auxiliary signal**. For technical discussion, the important distinction is between the learned statistical state and the deterministic decoder logic: the former contains Ridge coefficients and empirical counts; the latter encodes scene reconstruction, constraint propagation, and dynamic programming.

For a concise mental model:

```text
          TRAINING QUESTIONS + TRAINING IR DURATIONS
                           │
                           ▼
                   structured knowledge
                           │
          ┌────────────────┴─────────────────┐
          │                                  │
          ▼                                  ▼
   text / scene decoder              Ridge duration learner
          │                                  │
          └────────────────┬─────────────────┘
                           ▼
                  hypothesis scoring
                           │
                           ▼
                    final predictions
```

**Key takeaway:** the final system has **no neural model weights**. Its learned numeric component is a compact Ridge model plus deterministic empirical statistics; most of the performance comes from exploiting the benchmark's latent scene structure.
