# Work Split (2 members)

Member A = <Name 1, SRN>   |   Member B = <Name 2, SRN>

Rule: each person owns their files and writes the commits for them, but BOTH must be able to
explain everything (viva questions are asked to each person, and the demo is shared).

## Member A: Data + Classical ML (Perceptron / SVM)
| Task | File / output |
|---|---|
| Create private GitHub repo, add teammate, faculty, TAs | GitHub settings |
| Download Kaggle dataset, explore it (class counts, feature ranges) | `data/README.md`, notebook or notes |
| Preprocessing: cleaning, class balancing, split, scaling | `src/common.py` |
| Perceptron + SVM (RBF, sigmoid, poly), 5-fold CV, scaled vs unscaled | `src/train_baselines.py` |
| Tune C and gamma for RBF SVM, per-class report, confusion matrix | `results/svm_report.txt`, `svm_confusion.png`, `baseline_cv.csv` |
| Write-up sections: 1 Problem, 2 Dataset, 3 Approach (preprocessing + SVM part) | `docs/WRITEUP.md` |
| Slides: problem, dataset, preprocessing, classical models | slide deck |
| README: setup + dataset instructions | `README.md`, `requirements.txt` |

## Member B: Neural Network + Demo + Evaluation
| Task | File / output |
|---|---|
| Neural network (1 hidden layer, sigmoid, LogSoftmax, NLL) | `src/train_nn.py` |
| Experiment with hidden size, epochs, learning rate, L2; record what worked | `results/nn_report.txt`, notes |
| Loss and accuracy curves, confusion matrix | `results/nn_curves.png`, `nn_confusion.png` |
| Live demo script, tested from a fresh clone | `src/demo.py` |
| Final model comparison table (all 5 models) | `README.md` results table, `docs/WRITEUP.md` section 5 |
| Write-up sections: 3 Approach (NN part), 4 Implementation, 5 Results, 6 Conclusions | `docs/WRITEUP.md` |
| Slides: NN architecture, results, plots, demo, conclusion | slide deck |
| Fill team contributions, final checklist, export write-up to PDF | `docs/TEAM_CONTRIBUTIONS.md`, `writeup.pdf` |

## Shared (both)
- Review each other's code via pull request or at least read it before the review.
- Rehearse the presentation together; each presents their half; both attend the demo.
- Split the viva questions in `docs/VIVA_QA.md`: A studies Q1-6, 9, 12; B studies Q6-8, 10, 11. Then swap and quiz each other.

## Dependency order
A finishes `common.py` first (B needs it for the NN) -> B then trains NN -> B collects all numbers -> both finalize write-up and slides.
