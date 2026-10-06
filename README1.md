# Playlist Classifier: Assigning Songs to Playlists

UE24CS352A Machine Learning mini-project.
Team: <NEHA JOSHI 1 (PES1UG24CS297)>, <NIDHI L 2 (PES1UG24CS298)> | Section: <E> | Problem no.: <26>

Based on the paper *"Training a Playlist Curator Based on User Taste"* (Awadelkarim & Coelho, 2018).

## Problem
Given a song described by Spotify audio features, predict which playlist it belongs to
(multi-class classification). Each genre acts as one playlist.

## Models compared
Perceptron, SVM (RBF / sigmoid / polynomial kernels), and a 1-hidden-layer neural network (PyTorch).

## Setup
```bash
git clone <repo-url> && cd playlist-classifier
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Download the dataset as described in `data/README.md`.

## Run
```bash
cd src
python train_baselines.py      # Perceptron + SVMs, 5-fold CV, tuned RBF SVM
python train_nn.py             # neural network (options: --hidden 32 --epochs 300 --lr 0.01)
python demo.py --random        # live demo on a held-out song
python demo.py --features 0.6 0.8 5 -5.0 1 0.05 0.1 0.2 0.7 120
```
Outputs (metrics, confusion matrices, loss/accuracy curves) are written to `results/`.

## Project structure
```
src/common.py            data loading, class balancing, split, scaling
src/train_baselines.py   Perceptron + SVMs
src/train_nn.py          PyTorch neural network
src/demo.py              predict a playlist for a song
docs/                    write-up, slides outline, work split, commit plan, viva prep, checklist
results/                 generated plots and reports
```

## Results
Fill in after running (copy from `results/`):

| Model | Test accuracy |
|---|---|
| Perceptron | |
| RBF SVM (tuned) | |
| Sigmoid SVM | |
| Poly SVM | |
| Neural network (1 hidden layer) | |

## Contributions
See `docs/TEAM_CONTRIBUTIONS.md` and `docs/WORK_SPLIT.md`.
