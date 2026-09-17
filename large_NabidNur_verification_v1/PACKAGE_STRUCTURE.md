# CUHK-X Verification Package Structure

This folder is a participant-facing template. Rename the top-level folder and replace every value marked `REPLACE_ME` before submission.

## Package layout

```text
small_or_large_TEAMNAME_verification_v1/
├── README.md
├── PACKAGE_STRUCTURE.md
├── inference.sh
├── manifest.yaml
├── src/
├── environment/
├── configs/
├── artifacts/
├── submission/
│   ├── README.md
│   ├── submission_info.yaml
│   ├── final_submission_1.csv
│   └── final_submission_2.csv          # optional; only if two were Selected
└── declarations/
    ├── CUHK-X_Honor_Declaration.docx
    └── README.md
```

The internal layout of `src/` is flexible. Do not rename `README.md`, `inference.sh`, or `manifest.yaml`, and keep all paths in the package relative to the package root. `PACKAGE_STRUCTURE.md` is an instructional reference supplied by the Organizing Committee; teams may leave it unchanged.

## Packaging by track

Submit one package per track. A team shortlisted in both tracks must submit two separate packages, one for `small` and one for `large`.

- **Small Model Track:** submit the complete folder as one ZIP archive. Put the final deployable model in `artifacts/`. The published 100 MB model limit still applies.
- **Large Model Track:** a single ZIP is not mandatory when model artifacts are too large. Upload the code and documents as one ZIP and upload large weight files as separate files in the same submission form (up to 10 files per submission and 10 GB per file; split larger files into parts of at most 9 GB). List every separate file in `manifest.yaml`.
- **Public base model plus adapter:** include the adapter and all custom artifacts. A public base model need not be duplicated if its repository, exact revision, required files, licence, and access method are recorded. Such downloads may happen during environment setup only — inference itself must run without network access unless declared.
- **API-only solution:** include complete prompts, parameters, executable client code, and an example configuration without secrets. Never include real API keys.
- **Hybrid or ensemble solution:** describe every component and the execution order in `README.md` and `manifest.yaml`.

## Environment: Docker preferred

A Docker image (Dockerfile plus a `docker save` archive or a pinned registry tag) is the preferred way to provide the environment; a pinned `requirements.txt` / `environment.yml` is accepted. A `docker save` archive may be uploaded as a separate file next to the ZIP. See `README.md` section 4 for the command we will use.

## Execution contract

The Organizing Committee will run the package from its root directory using:

```bash
bash inference.sh <data_dir> <output_csv>
```

This two-argument command must reproduce the primary verification submission. When two Selected submissions used different models or configurations, the same entry point must additionally support:

```bash
bash inference.sh <data_dir> <output_csv> <submission_id>
```

The primary verification submission is always run. A second Selected submission is normally subject to static audit only, but it must remain runnable through the same entry point if the Organizing Committee requests an escalated check.

The Organizing Committee will choose the local server path passed as `<data_dir>`. This path will contain the relevant track's official label-free `Testing` data in the same internal layout as the published challenge dataset. The script must use that supplied path, include the preprocessing used by the target submission, and write a submission-format CSV exactly to `<output_csv>`. It must not depend on an absolute path from the participant's computer or require Training labels or test ground truth. Any network, API, system-package, custom CUDA, or licence requirement must be declared before execution.

## Before submission

1. Extract or copy the package into a clean directory.
2. Follow `README.md` without relying on undocumented manual steps.
3. Run `inference.sh` using the published official label-free `Testing` directory for the relevant track, or an organizer-approved test-format subset with the same internal layout.
4. Confirm that the output CSV has the correct schema and row order.
5. Include every submission marked `Selected` on Kaggle for this track, up to two. Obtain each CSV using its Kaggle Submission ID and do not edit or rewrite its contents.
6. Complete `submission/submission_info.yaml`, map each Submission ID to its model/checkpoint/configuration, and enter the `primary_verification_submission_id` confirmed by the Organizing Committee.
7. Remove credentials, private keys, caches, temporary files, and unnecessary datasets.
8. Complete and sign `declarations/CUHK-X_Honor_Declaration.docx`, and keep the signed declaration in the same directory in the submitted package.
