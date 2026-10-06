# Commit Plan (shows steady, shared work in the repo history)

Use branches if you like: `member-a/baselines`, `member-b/nn`, merge to `main` via pull request.
Each member commits from their own GitHub account/git config (`git config user.name`).

## Tue 6 Oct
- A: `git init`, create repo, add collaborators. Commit: "Initial project structure, README, requirements"
- A: Commit: "Add dataset instructions and data folder"
- A: Commit: "Add common.py: data loading, balancing, split, scaling"
- B: Clone, set up environment. Commit: "Add docs templates (writeup, checklist, viva prep)"

## Wed 7 Oct
- A: Commit: "Add Perceptron and SVM baselines with 5-fold CV"
- A: Commit: "Tune RBF SVM, add classification report and confusion matrix"
- B: Commit: "Add PyTorch neural network training script"
- B: Commit: "Add loss/accuracy curves and NN evaluation report"

## Thu 8 Oct
- B: Commit: "Tune NN hyperparameters (hidden size, epochs, L2)"
- B: Commit: "Add live demo script (random song and manual features)"
- A: Commit: "Add results from baseline experiments to results/"
- A: Commit: "Fill write-up sections 1-3 (problem, dataset, preprocessing)"

## Fri 9 Oct
- B: Commit: "Add model comparison table to README and write-up"
- B: Commit: "Fill write-up sections 4-6 and add writeup.pdf"
- A: Commit: "Add presentation slides (problem, data, classical models)"
- B: Commit: "Add presentation slides (NN, results, demo)"
- Both: Commit: "Update team contributions, final README"

## Sat 10 Oct (deadline 11:59 PM, aim to finish early)
- Fresh-clone test: clone into a new folder, follow README exactly, fix any gaps. Commit fixes.
- Tick `docs/CHECKLIST.md`. Final commit: "Final submission".

Note: reviews run 5-9 Oct. Ask your faculty/TA for your slot today and work backwards, because the code and demo must work BEFORE your review, not only before the deadline.

## Commit tips
- Small, descriptive messages. No single giant commit.
- Never commit `data/dataset.csv` or `artifacts/` (already in `.gitignore`).
- Commands: `git add .` -> `git commit -m "message"` -> `git push`. Run `git pull` before starting work.
