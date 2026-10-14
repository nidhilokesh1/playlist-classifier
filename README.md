# Playlist Classifier: Assigning Songs to Playlist Classes

UE24CS352A Machine Learning Mini-Project

**Team:** Nehaa Joshi (PES1UG24CS297), Nidhi L (PES1UG24CS298)  
**Section:** E  
**Problem No.:** 26

Based on the paper *"Training a Playlist Curator Based on User Taste"* by Amel Awadelkarim and Kevin Coelho.

## 1. Problem Statement

Playlist curation requires assigning songs to suitable groups based on their characteristics. This project formulates playlist assignment as a supervised multi-class classification problem. Given a song represented by Spotify audio features, the model predicts a genre-based class.

The publicly available dataset used in this project provides music genre labels rather than user-created playlists. Therefore, **genres are used as proxy playlist classes**. This is a classification adaptation, not a personalized recommendation system.

The project compares classical machine-learning models with a one-hidden-layer neural network.

## 2. Dataset

The project uses the Spotify Tracks Dataset. Each song is represented using these ten audio features:

- `danceability`
- `energy`
- `key`
- `loudness`
- `mode`
- `speechiness`
- `acousticness`
- `liveness`
- `valence`
- `tempo`

The target variable is `track_genre`. Twelve genres are used as classes:

`ambient`, `blues`, `classical`, `country`, `gospel`, `heavy-metal`, `hip-hop`, `jazz`, `opera`, `punk`, `reggae`, and `techno`.

After removing tracks with ambiguous genre assignments and rows with missing feature values, the selected data contains **1,737 usable samples**. Up to 150 samples per class are used; reggae has 87 usable samples and each of the other classes has 150.

## 3. Preprocessing and Evaluation

The ten audio features have different numerical scales, so they are standardized with `StandardScaler`. The scaler is fitted only on the training data and then applied to the test data to avoid data leakage.

Tracks whose IDs appear under multiple genres are removed because they have ambiguous target labels. The data is divided using an 80:20 stratified train/test split with random seed `42`. The held-out test set contains 348 samples.

## 4. Models

The evaluated models are:

- Perceptron
- SVM with RBF kernel
- SVM with sigmoid kernel
- SVM with polynomial kernel
- Tuned RBF SVM
- One-hidden-layer neural network

The RBF SVM is tuned using five-fold cross-validation on the training split. The grid searches `C` over `{0.1, 1, 10, 100}` and `gamma` over `{scale, 0.01, 0.1, 1}`.

### Neural Network

The final neural network uses:

- Input: 10 audio features
- Hidden layer: 64 neurons with sigmoid activation
- Output: 12 classes with `LogSoftmax`
- Loss: Negative Log Likelihood Loss
- Optimizer: Adam
- Learning rate: `0.01`
- Weight decay: `0.0001`
- Training epochs: `300`

The hidden-layer size was selected experimentally:

| Hidden neurons | Test accuracy |
|---:|---:|
| 12 | 48.6% |
| 32 | 50.0% |
| **64** | **53.2%** |
| 128 | 48.3% |

The 64-neuron configuration produced the best test accuracy among the tested neural-network configurations.

## 5. Results

All models were evaluated on the same held-out test set of 348 samples using the shared data-loading and preprocessing pipeline.

| Model | Test accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Perceptron | 34.20% | 0.2951 | 0.3053 |
| RBF SVM | 48.28% | 0.4616 | 0.4674 |
| Sigmoid SVM | 42.82% | 0.4099 | 0.4177 |
| Polynomial SVM | 48.56% | 0.4818 | 0.4817 |
| Tuned RBF SVM | 48.28% | 0.4738 | 0.4719 |
| **Neural Network (64 hidden neurons)** | **53.20%** | **0.51** | **0.52** |

The neural network achieved the highest test accuracy among the evaluated models, exceeding the polynomial SVM by 4.64 percentage points. The polynomial SVM was the strongest classical model by test accuracy and macro F1.

Performance varies across genre classes. In the neural-network evaluation, heavy-metal, hip-hop, and techno achieved F1 scores of approximately 0.71, while blues achieved 0.21. The neural network's training accuracy was 65.0%, compared with 53.2% on the test set, so some overfitting may be present.

These findings apply to the selected dataset, ten audio features, twelve genre-based classes, and fixed split. Since genre labels are proxies for playlist classes, the experiment does not directly demonstrate personalized playlist recommendation. The reference paper's reported results are not directly comparable because its data and task differ.

## 6. Implementation

The project is implemented in Python. Main libraries include:

- pandas and NumPy
- scikit-learn
- PyTorch
- matplotlib
- joblib

### Project Structure

The repository contains the following source files and committed results. The local dataset CSV and generated model artifacts are excluded from Git.

```text
playlist-classifier/
├── data/
│   └── README.md
├── docs/
│   ├── CHECKLIST.md
│   ├── COMMIT_PLAN.md
│   ├── PRESENTATION_OUTLINE.md
│   ├── TEAM_CONTRIBUTIONS.md
│   ├── VIVA_QA.md
│   ├── WORK_SPLIT.md
│   └── WRITEUP.md
├── results/
│   ├── baseline_test_results.csv
│   ├── perceptron_test_confusion.png
│   ├── perceptron_test_report.txt
│   ├── polynomial_svm_test_confusion.png
│   ├── polynomial_svm_test_report.txt
│   ├── rbf_svm_test_confusion.png
│   ├── rbf_svm_test_report.txt
│   ├── sigmoid_svm_test_confusion.png
│   ├── sigmoid_svm_test_report.txt
│   ├── tuned_rbf_svm_test_confusion.png
│   ├── tuned_rbf_svm_test_report.txt
│   ├── nn_confusion.png
│   ├── nn_curves.png
│   ├── nn_experiments.txt
│   └── nn_report.txt
├── src/
│   ├── common.py
│   ├── data_prep.py
│   ├── demo.py
│   ├── evaluate_baselines.py
│   ├── train_baselines.py
│   └── train_nn.py
├── README.md
└── requirements.txt
```

## 7. Setup and Running

Use Python and install the project dependencies in a virtual environment:

```bash
python -m venv .venv
```

Activate the environment, then install requirements:

```bash
# Windows Git Bash
source .venv/Scripts/activate
pip install -r requirements.txt
```

Place the Spotify dataset CSV at the location expected by `src/common.py`. The local `data/dataset.csv` file is not committed to the repository. Check `data/README.md` for dataset notes.

Run the model scripts from the repository root:

```bash
python src/train_nn.py
python src/evaluate_baselines.py
python src/demo.py --random
```

The baseline evaluation script writes a consolidated table to `results/baseline_test_results.csv` and generates per-model test reports and confusion matrices. Generated model artifacts are stored locally and are excluded from Git.

## 8. Team Contributions

**Nehaa Joshi (PES1UG24CS297):** implemented and evaluated neural-network configurations; recorded final neural-network metrics and experiments; added a runnable demo; performed classical-model held-out evaluation and produced baseline reports/confusion matrices; consolidated model comparisons and updated the README.

**Nidhi L (PES1UG24CS298):** contributed to the initial project setup and initialization, created and organized the required project files, committed project files to the GitHub repository, and assisted with project documentation and final deliverable preparation.
