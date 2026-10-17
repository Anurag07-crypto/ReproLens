# ReproLens demo (about 2 minutes)

1. Start with the question: “Does the code actually match the paper?”
2. Choose `iTransformer` in the repository dropdown.
3. Open a supported claim and show the verdict, explanation, source file, and exact line numbers.
4. Open a contradicted claim and explain the difference between the paper statement and the code evidence.
5. Point out that “not found” means the retrieved repository evidence did not establish the claim, not that the feature is definitively absent.
6. End with the dev/test evaluation table. Use only measured numbers from `eval/results.md`; do not quote placeholder or empty-split results.
7. Close with the limitation: ReproLens checks specific paper claims against code and does not rerun experiments or certify reproducibility.
