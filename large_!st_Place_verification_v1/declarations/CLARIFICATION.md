# Clarification accompanying the Honor Declaration - team Nabid Nur, Large Model Track

We want the Organizing Committee to have the complete picture before evaluating our package. Our signatures on the
Honor Declaration are given together with this clarification.

1. **Answers set by hand.** The Selected submissions contain answers that were set by hand rather than produced by the
   model code:
   - **Submission 56255655:** `test_0358`, `test_0681` and `test_0423`.
   - **Submission 56261134:** the same three plus `test_0430`.
   - **Five earlier answers:** `test_0331`, `test_0333`, `test_0340`, `test_0355` and `test_0641` were first set by
     hand with an AI assistant, during development of submission v3. The stage-2 slot rule, written afterwards and
     validated on held-out training users, now reproduces them.

   Every row is listed with an explanation in `artifacts/disclosed_adjustments_<id>.json`. `configs/model_only.yaml`
   shows the model's own output.
2. **Declaration statement "We have not manually labelled test samples".** Read strictly, it is not fully true for
   us, because of the answers in point 1. We never had or used test ground-truth labels, and no test answer was used
   to train the model state.
3. **Declaration statement "the frozen components used for those submissions".** The code in `src/` consolidates,
   after the deadline, the research scripts that produced the submissions (README section 2.5). It reproduces both
   submitted CSVs byte for byte, and the method was not changed.
4. **Public-leaderboard use.** Public-leaderboard feedback determined several decisions. The main one is the repeat
   indices of two-clip scenes, found with a deliberate probe submission. It also chose the hypothesis for scene
   LM_test_0158-0160 and `test_0430` in 56261134 (README section 2.4).
5. **AI assistance.** Claude (Anthropic) agents read the label-free test questions and proposed answers for individual
   test rows during development. For submission v5, which is not Selected, LLM judges assigned probabilities to
   answer hypotheses for test questions. No LLM or API is used at inference.
6. **Nature of the method.** The method relies on how this test set was constructed: re-performed training scene
   scripts, the recording order of clips, and the three-manner scene structure. We expect much lower accuracy on data
   built differently.

We are happy to answer any question at the contact address in `manifest.yaml`.
