# Evaluation results

Metrics are computed only for manually labelled claims assigned to each split. An empty split is not a measured result and is reported as n/a.

| Split | System | Claims | Accuracy | Supported P/R | Contradicted P/R | Not found P/R |
|---|---|---:|---:|---:|---:|---:|
| dev | retrieval | 6 | 0.500 | 1.000/0.500 | 0.000/0.000 | 0.333/1.000 |
| dev | baseline | 6 | 0.333 | 1.000/0.250 | 0.000/0.000 | 0.200/1.000 |
| test | retrieval | 6 | 0.333 | 0.500/0.500 | 0.000/0.000 | 0.333/1.000 |
| test | baseline | 6 | 0.167 | 0.000/0.000 | 0.000/0.000 | 0.200/1.000 |

Precision and recall are one-vs-rest, per verdict. Empty splits are reported as n/a.
