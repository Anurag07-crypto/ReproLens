# Submission summary

ReproLens checks whether a research paper’s experimental claims are supported by its GitHub repository. It extracts checkable details such as datasets, model architecture, hyperparameters, training setup, and preprocessing, then searches line-numbered repository code and returns a verdict with a reason and source citations.

Unlike a standalone LLM response, each verdict is grounded in retrieved code evidence. The evaluation scripts compare retrieval-based predictions to a retrieval-free baseline on separately labelled development and test repositories. Report measured results only after populating the hand-labelled dataset and running the evaluation.

ReproLens checks claims against code; it does not rerun experiments or verify reproducibility.
