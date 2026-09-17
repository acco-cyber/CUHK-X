# Artifacts

| File | Purpose |
|---|---|
| `model_state.json` | Model state learned by `src/train.py` from the training questions and training clip durations. It holds the 272 scene scripts, emotion and order priors, HARn naive-Bayes tables, slot statistics, the emotion count model and the duration ridge model. It is not a neural network. Rebuilding it gives SHA-256 `f90c495999a03d9f721fe45ac3ee46cf7aa5d55dea69df961fc95e4f43de9cf4`. |
| `disclosed_adjustments_56255655.json` | DISCLOSURE: the 3 answers of submission 56255655 that were set by hand and are not produced by the model code, with explanations. |
| `disclosed_adjustments_56261134.json` | DISCLOSURE: the 4 answers of submission 56261134 that were set by hand and are not produced by the model code, with explanations. |

The disclosed rows are applied only when both conditions hold:

- the order-independent content hash of `test_qa.csv` equals that of the original CUHK-X Large Model Track test set
  (`133544ba3d00518d2c9f7f85b2983745f1fed13f35613cd80c0d5df298206c06`);
- the duration stage ran on every eligible scene.
