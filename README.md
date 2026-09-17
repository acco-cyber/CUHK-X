# Team Nabid Nur - CUHK-X Large Model Track - Reproduction Guide

Structure-aware multiple-choice decoder for the CUHK-X Large Model Track (VQA). It is **pure Python standard library**
and uses **no GPU, no network, no API and no neural checkpoint**. A run over the test set takes a few seconds on one
CPU core.

![architecture](Architecture_Diagram_of_V6)

## 0 Needs attention (answer first)

- More than one GPU needed: `no` (no GPU is used at all)
- More than 24 GB of GPU memory needed: `no` (peak: `0 GB`)
- Specific GPU generation or driver needed: `no`
- Network access needed at inference time: `no`
- Runs on Linux x86_64: `yes`. It needs CPython 3 and bash, and was tested with CPython 3.12.13 on Linux and 3.11.9
  on Windows. No prebuilt Docker image is supplied (section 4).
- **Please read sections 2.4 and 2.5 before evaluating.**
  - Each Selected submission contains answers that were set by hand and are not produced by the model code: 3 rows
    in 56255655 and 4 rows in 56261134.
  - Several decisions were determined with public-leaderboard feedback.
  - The code in `src/` is a consolidation, written after the deadline, of the research scripts that produced the
    submissions. It reproduces both submitted CSVs byte for byte.

## 1 Team and submission

- Team name: `Nabid Nur` (Kaggle members: koushikrudra, nabidnur)
- Track: `large`
- Selected Kaggle Final Submission record: `submission/submission_info.yaml`
- Selected final submissions (to be confirmed by the team on Kaggle > Submissions > Selected):
  - #1: Kaggle submission ID `56255655`, file name `text_decoder_v6.csv`, submitted on `2026-09-15T13:41:34Z`,
    public score `0.99707`, private score `0.98529`, SHA-256 of the exact submitted CSV
    `c2792401d8fc4c790d92252e537f362365a690052199e263a370c8713577cb1d`
  - #2: Kaggle submission ID `56261134`, file name `text_decoder_v7A.csv`, submitted on `2026-09-15T18:52:29Z`,
    public score `0.99707`, private score `0.98529`, SHA-256
    `e9bead9015e68442ab6db3ca2007ee2af5d018f5abf4327c21704074fd66e2f0`
- Primary verification target: `#1 (56255655)`, proposed by the team.
  - Both Selected submissions have the same private score (0.98529). #1 is the earlier one and contains fewer
    disclosed rows; the two differ in exactly one answer (`test_0430`).
  - Kaggle's private leaderboard shows the team with the timestamp of #2 (2026-09-15 18:52:29 UTC). If the Committee
    confirms #2 as the primary target, `bash inference.sh <data_dir> <output_csv> 56261134` reproduces it byte for
    byte.
- Solution type: `local_checkpoint`. The checkpoint is a JSON model state learned from the training questions and
  training clip durations; there is no neural network.

## 2 Frozen solution summary

### 2.1 Idea

The test clips are recordings of daily-activity *scene scripts*. Each script is a short list of actions performed
three times, once per manner adverb (typically slow, steady, then hurried). Every question about a clip is a
multiple-choice view of that script. The model therefore decodes all questions of a scene jointly:

1. **Scene grouping.** Clips are processed in the order of the trailing number of their directory name
   (`LM_test_NNNN`). Consecutive clips whose emotion options share the three scene adverbs form one scene; their
   position inside the scene gives the repeat index. Two-clip scenes are repeats 2 and 3.
2. **Script matching.** Each scene is scored against the 272 training scene scripts: action set, adverb per repeat,
   and the precedence closure of its actions. Several test participants re-performed scripts that occur in the
   training set. A matched script supplies the scene's action set, the emotion for each repeat and the action order.
3. **Constraint decoding.** The training questions show strong regularities:
   - a single-question distractor is never a scene action (1 exception in 2,427);
   - a false multi-select option is never a scene action (1 in 1,729);
   - exactly one combination option consists only of scene actions (789 of 790).

   Answers are decoded so that every question of a scene agrees with one action set.
4. **Clip duration.** For scenes without a matched script, the emotion question asks which repeat was performed in
   which manner, and the hurried repeat is the shortest recording. Duration is read from the MP4 movie header of the
   clip's IR video; no pixels are decoded.

### 2.2 Stages (execution order, `src/cuhkx_vqa/pipeline.py`)

| Stage | Module | What it does | Validation evidence |
|---|---|---|---|
| 1 | `text_decoder.py` | Scene grouping, script matching and joint decoding of all HAU questions. For HARn clips, whose ids are sorted by label: a softly monotone label dynamic programme with naive-Bayes option scores. | Held-out training users. Twin scenes: 3 repeats 0.971; 2 repeats 0.87-0.99 depending on which repeat is missing. Scenes without a script: 3 repeats 0.863. HARn: 0.994. |
| 2 | `slot_rule.py` | Temporal-order questions whose action pairs are missing from the matched script. Actions that substitute for each other across repeats share a slot. The best decoder-scored order among the consistent ones is chosen. | Held-out training users: 42 of 43 missing pairs. |
| 3 | `scene_joint.py` | Scenes without a matched script: single and combination answers are chosen jointly under the distractor constraints, and multi answers follow from the resulting action set. | Held-out training users. 3-repeat scenes: 51 answers fixed, 0 broken. 2-repeat scenes: 52 fixed, 3 broken. |
| 4 | `singleton.py` | One-clip scenes: a count model of P(repeat \| adverb) with a manner lexicon, marginalised over repeats 2 and 3. | Held-out training users: 0.497 vs 0.405. |
| 5 | `duration.py` | Scenes of 2-3 clips without a matched script: a ridge model of within-scene log duration from adverb class, adverb and repeat. The assignment of the scene's adverbs maximises the repeat prior plus the duration likelihood (weight 1). | Held-out training users (leave-one-group-out): 3 repeats 0.819 to 0.922; 2 repeats 0.760 to 0.913. Test scenes with a matched script, decoded without it and scored against the script's adverbs (not ground truth): 0.796 to 0.959, or 0.939 with the script owners' training clips removed. That test-scene check was also used when fixing the weight at 1. |
| 6 | `pipeline.py` | **Disclosed rows** (section 2.4). Applied only when the canonical content hash of `test_qa.csv` equals that of the original test set and stage 5 ran on every eligible scene. | Not a model stage. |

On the original test set the stages change: 5 answers in stage 2, 1 in stage 3, 1 in stage 4 and 13 in stage 5.
Training (`src/train.py`) builds `artifacts/model_state.json` deterministically from `Training/training_qa.csv` and
the IR video durations in `Training/data/HAU.zip`.

### 2.3 Mapping to the Selected submissions

| Submission | Config | Model state | Stages | Disclosed rows |
|---|---|---|---|---|
| 56255655 (primary) | `configs/final.yaml` | `artifacts/model_state.json` | 1-6 | `artifacts/disclosed_adjustments_56255655.json` (3 rows) |
| 56261134 | `configs/submission_56261134.yaml` | same | 1-6 | `artifacts/disclosed_adjustments_56261134.json` (4 rows) |
| (reference) | `configs/model_only.yaml` | same | 1-5 | none |

### 2.4 Disclosure (please read)

- **Answers set by hand.** `artifacts/disclosed_adjustments_<id>.json` lists every submitted answer that the model
  code does not produce, with an explanation:
  - `test_0358`: a temporal-order answer set by hand, with an AI assistant, during development.
  - `test_0681` and `test_0423`: in scene LM_test_0158-0160 the duration model's two best hypotheses were nearly tied
    (0.53 vs 0.46). The team chose the 0.46 hypothesis because it best explains the public score of an earlier
    submission (v5).
  - `test_0430` (56261134 only): changed from the stage-4 answer using duration and bitrate evidence plus a
    consistency analysis over the team's public-leaderboard scores.

  Without these rows, the model's output differs from 56255655 on exactly 3 rows and from 56261134 on exactly 4.
- **Leaderboard use during development.** Kaggle lists 31 team submissions: 23 in August from an earlier, different
  pipeline that is not part of this package, and 8 on 14-15 September with this method. Public-leaderboard feedback
  *determined*, not merely confirmed, these decisions:
  1. **`two_repeat_positions = [2, 3]`.** v2 (repeats 1 and 3, public 0.95321) and the deliberate probe v2_map12
     (repeats 1 and 2, 0.92397) differ only in 21 emotion answers. Together they showed that the second clip of a
     two-clip scene is repeat 3; v3 (0.98245) then used repeats 2 and 3. That the first clip of each three-clip scene
     is listed last in `test_qa.csv` was noticed only after these probes.
  2. The hypothesis chosen for scene LM_test_0158-0160.
  3. `test_0430` in 56261134.
  4. Keeping the six hand-set temporal-order answers after v3's public gain.

  Hyperparameters of stages 1-5 were chosen on held-out training users.
- **AI assistance in development.** The team used Claude (Anthropic) as a coding and analysis assistant, including
  multi-agent workflows of Claude sub-agents. These agents did three things:
  - read the training data and the label-free test questions;
  - wrote and validated the stage 2-5 rules;
  - proposed answers for individual test rows: the six temporal-order answers of v3, `test_0358` among them, and the
    two v4 row changes, which came from the coded solvers now in stages 3 and 4.

  For v5 (56255194, not Selected), Claude "judge" agents also assigned probabilities to answer hypotheses for test
  questions after being scored on held-out training look-alikes. No judge output is in either Selected submission.
  No LLM, API or network service is used at inference.
- **Data.** No test ground-truth labels were used, and no external dataset or pretrained model. Clip durations come
  from the official CUHK-X test and training videos, downloaded with HTTP range requests from the organisers' public
  Google Drive mirror linked on the challenge website.
- **Generalisation warning.** The model depends on properties of how this test set was constructed:
  - participants re-performing training scene scripts;
  - clips ordered by participant, scene and repeat;
  - the three-manner scene structure.

  On data without these properties, stage 1 falls back to text priors and accuracy will be far lower than on the
  Kaggle test set. The code still runs on such data and writes a valid CSV; see section 8.

### 2.5 Provenance of the code (please read)

The Kaggle CSVs were produced during the competition by research scripts, not by `src/cuhkx_vqa/`:

- **v3 (56236009):** the research text decoder, with two-clip scenes mapped to repeats 2 and 3 (section 2.4), plus
  six temporal-order answers set by hand with an AI assistant.
- **v4 (56253165):** v3 plus two row changes, `test_0079` and `test_0430`. They were proposed by coded solvers, a
  joint scene-consistency decoder and a singleton emotion count model, which were validated on held-out training
  users, and applied as a change list.
- **v6 (56255655):** v4 plus the emotion answers of the duration model, with scene LM_test_0158-0160 set to its
  second-ranked hypothesis.
- **v7A (56261134):** v6 plus `test_0430` A.

On 16 September 2026, after the deadline, the research scripts were consolidated into `src/` without changing the
method:

- The decoder, the two solvers and the duration model were ported line by line.
- The stage-2 slot rule was written after the six hand-set answers existed. It was validated on held-out training
  users (42 of 43 missing pairs) and reproduces five of those answers.
- Every remaining difference from the Selected CSVs is listed in the disclosed-rows files.

## 3 Hardware and runtime

- Operating system: tested on Windows 11 (CPython 3.11.9, Intel Core i7-1265U, 31 GB RAM) and on Linux x86_64 (glibc
  2.35, CPython 3.12.13, Kaggle CPU session)
- GPU model or minimum compute capability: none
- GPU count: 0
- Minimum GPU memory per GPU: 0 GB
- Minimum host memory: 1 GB
- Free disk space required: 10 MB for the package, plus the test data supplied by the organisers
- Expected environment setup time: none if Python 3 and bash are installed
- Expected inference time: about 1-1.5 seconds of decoding, measured on both machines, plus reading and inflating
  the 208 IR videos from the test zip, estimated at a few seconds. A full run on the complete official zip has not
  been timed; see section 7. The zip does not need to be extracted.

## 4 Environment setup

**Option B (no Docker; recommended for this package).** Any Linux x86_64 machine with Python 3 and bash. There are no
third-party packages, system libraries, CUDA, compilers or environment variables:

```bash
sudo apt-get update && sudo apt-get install -y python3     # only if python3 is missing
bash inference.sh <data_dir> <output_csv>
```

`environment/requirements.txt` is intentionally empty.

**Optional Docker.** `environment/Dockerfile` is provided, but no image has been built or tested by the team.
Building it needs network access to pull `python:3.11.9-slim-bookworm`; running it does not:

```bash
docker build -f environment/Dockerfile -t nabidnur/cuhkx-large-vqa:v1 .
docker run --rm --network none -v <data_dir>:/data:ro -v <output_dir>:/output \
    nabidnur/cuhkx-large-vqa:v1 bash inference.sh /data /output/submission.csv
```

## 5 Run inference

```bash
bash inference.sh <data_dir> <output_csv>                 # reproduces the primary submission 56255655
bash inference.sh <data_dir> <output_csv> 56261134        # reproduces the second Selected submission
python3 src/predict.py --data-dir <data_dir> --output-csv <output_csv> --config configs/model_only.yaml   # model only
```

Expected files below `<data_dir>`, the official label-free `Testing` directory:

- **`test_qa.csv`.** The shallowest `test_qa.csv` at most 4 directory levels below `<data_dir>` is used. Several at
  the same depth is an error: pass the Testing directory itself.
- **Clip videos.** They are found either as extracted files (`.../<clip dir>/IR/IR.mp4`) or inside any `.zip` below
  `<data_dir>`, for example `data/large_model_track_test.zip`.
  - Only the MP4 header of the `IR` video is needed. Per scene, one modality is used for all clips, in the order IR,
    Thermal, Depth, Depth_Color.
  - Unreadable archives and videos are logged as warnings and skipped. An affected scene keeps its text-decoder
    answers.

Output: `<output_csv>` has columns `qa_id,prediction`, one row per `test_qa.csv` row in the same order, UTF-8 with
`\r\n` line endings (the Python `csv` default). Progress is logged to stderr (`[stage1]` to `[stage6]`, plus
`WARNING` lines). A fatal input problem, such as a missing `test_qa.csv` or missing columns, exits non-zero with a
message. Any prediction with an invalid format is replaced by a valid default and counted in the log (0 on the
official test set).

## 6 Model artifacts and external resources

| Artifact | Purpose | Size | SHA-256 |
|---|---|---|---|
| `artifacts/model_state.json` | training-derived model state (stages 1-5) | 151,483 B | `f90c495999a03d9f721fe45ac3ee46cf7aa5d55dea69df961fc95e4f43de9cf4` |
| `artifacts/disclosed_adjustments_56255655.json` | disclosed rows of submission 56255655 | see `CHECKSUMS.sha256` | see `CHECKSUMS.sha256` |
| `artifacts/disclosed_adjustments_56261134.json` | disclosed rows of submission 56261134 | see `CHECKSUMS.sha256` | see `CHECKSUMS.sha256` |

There are no public base models, tokenizers, adapters, prompts or APIs at inference. Rebuild the model state with
the command below; with `--durations-json` it was verified to rebuild the shipped file byte for byte.

```bash
python3 src/train.py --training-dir <Training_dir>        # reads training_qa.csv and data/HAU.zip (HARn.zip is ignored)
```

## 7 Reproduction expectations

- **Determinism.** The output is exactly deterministic: no random seeds, pure-Python integer and float arithmetic.
  `inference.sh` sets `PYTHONHASHSEED=0`.
- **Reproduction runs.** Byte-identical reproduction of both Kaggle CSVs (SHA-256 `c2792401...` and `e9bead90...`)
  was verified on two platforms:
  - Windows 11, CPython 3.11.9;
  - Linux x86_64, glibc 2.35, CPython 3.12.13, in a private Kaggle CPU session.

  Both runs used the official `test_qa.csv` with a stand-in test zip whose `IR.mp4` members carry the original
  `mvhd` duration headers. The complete official zip could not be downloaded again: the Drive mirror's download
  quota was exceeded.
- **Real videos.** The MP4 reader was checked on the real official zip's first gigabyte. All 108 IR videos there give
  exactly the durations used for the submissions.
- **Entry point.** The runs called `src/predict.py` with the arguments `inference.sh` passes. `bash inference.sh`
  and the Docker image were not executed by the team; there is no bash on the Windows machine.
- **Ties.** Three answers come from exact or last-bit score ties between two candidates: `test_0343`, `test_0647`
  (temporal order) and `test_0509` (HARn). Evaluation order breaks them deterministically, and the tie scores were
  bit-identical on both platforms. If another platform ever disagreed, only these rows could change.
- **Other checks:**
  - `python3 tests/test_components.py` covers the MP4 reader in zip and folder layouts, format validation and the
    content hash.
  - `configs/model_only.yaml` equals the research pipeline's output on all 682 rows.
  - Robustness was checked on extracted, nested, shuffled, subset, renamed-id, video-less and corrupt-video inputs:
    no crash, valid CSV in input order.

## 8 Known limitations and troubleshooting

- **`[stage6] disclosed adjustments NOT applied`.** The content of `test_qa.csv` differs from the original test set,
  or stage 5 could not read every eligible scene's videos. The output then equals the model-only output for that
  input.
- **`[stage5] WARNING: no readable video durations ...`.** Videos are missing or unreadable, so the output will not
  match the Kaggle submission. Check that the test zip or the extracted clip folders are below `<data_dir>`.
- **`[pipeline] WARNING: ... clip directories have no trailing number`.** Scene grouping relies on recording order.
- **Line endings.** Windows line endings in `inference.sh` would break bash; the package ships LF line endings.

## 9 Final checklist

- [ ] All `REPLACE_ME` fields have been replaced (team contact email in `manifest.yaml` still to be added).
- [x] `manifest.yaml` matches the submitted files.
- [ ] The package runs from a clean directory through `inference.sh` (verified through `src/predict.py` on Windows
      and Linux; `bash inference.sh` not yet executed).
- [ ] Every Selected Final Submission ID and its exact CSV are included and mapped to its configuration (to be
      confirmed against Kaggle's Selected filter).
- [ ] `primary_verification_submission_id` is the ID confirmed by the Organizing Committee.
- [x] External data, models, APIs, licences and access requirements are declared (sections 2.4 and 6).
- [ ] `declarations/CUHK-X_Honor_Declaration.docx` is completed and signed by every team member (see
      `declarations/CLARIFICATION.md`).
- [x] No API keys, passwords, personal credentials, official raw challenge data or unnecessary caches are included.
