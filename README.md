# Playlist Classifier: Assigning Songs to Playlist Classes

UE24CS352A Machine Learning Mini-Project

**Team:** Nehaa Joshi (PES1UG24CS297), Nidhi L (PES1UG24CS298)
**Section:** E
**Problem No.:** 26

Based on the paper *"Training a Playlist Curator Based on User Taste"* by Amel Awadelkarim and Kevin Coelho.

## 1. Problem Statement

Playlist curation requires assigning songs to suitable groups based on their characteristics. This project formulates playlist assignment as a supervised multi-class classification problem.

Given a song represented by Spotify audio features, the model predicts the most suitable playlist class.

The publicly available dataset used in this project provides music genre labels rather than user-created playlists. Therefore, **genres are used as proxy playlist classes** for the classification task.

The project compares classical machine-learning models with a one-hidden-layer neural network.

## 2. Dataset

The project uses the Spotify Tracks Dataset.

Each song is represented using the following 10 audio features:

- danceability
- energy
- key
- loudness
- mode
- speechiness
- acousticness
- liveness
- valence
- tempo

The target variable is `track_genre`.

Twelve genres are used as the playlist classes:

- ambient
- blues
- classical
- country
- gospel
- heavy-metal
- hip-hop
- jazz
- opera
- punk
- reggae
- techno

After removing tracks with ambiguous genre assignments and missing feature values, the selected dataset contains approximately **1,737 usable samples**.

Up to 150 samples per class are used.

The data is divided into training and test sets using an 80:20 stratified split with a fixed random seed.

## 3. Preprocessing

The 10 audio features have different numerical scales. For example, loudness is measured in decibels and tempo is measured in BPM, while several other features lie between 0 and 1.

Therefore, the features are standardized using `StandardScaler`.

The scaler is fitted only on the training data and then applied to the test data to avoid data leakage.

Tracks that occur under multiple genres are removed because they would provide ambiguous target labels.

## 4. Models

The following models are evaluated:

- Perceptron
- SVM with RBF kernel
- SVM with sigmoid kernel
- SVM with polynomial kernel
- One-hidden-layer neural network

The neural network follows the general architecture described in the reference paper.

### Neural Network

The final neural network contains:

- Input layer: 10 audio features
- Hidden layer: 64 neurons
- Hidden activation: Sigmoid
- Output layer: 12 classes
- Output activation: LogSoftmax
- Loss: Negative Log Likelihood Loss
- Optimizer: Adam
- Learning rate: 0.01
- Weight decay: 0.0001
- Training epochs: 300

The hidden-layer size was selected experimentally.

| Hidden neurons | Test accuracy |
|---:|---:|
| 12 | 48.6% |
| 32 | 50.0% |
| **64** | **53.2%** |
| 128 | 48.3% |

The 64-neuron configuration produced the best test accuracy and was therefore selected as the final neural-network configuration.

## 5. Results

All models were evaluated on a held-out test set of 348 samples using the shared data-loading and preprocessing pipeline.

| Model | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Perceptron | 34.20% | 0.2951 | 0.3053 |
| RBF SVM | 48.28% | 0.4616 | 0.4674 |
| Sigmoid SVM | 42.82% | 0.4099 | 0.4177 |
| Polynomial SVM | 48.56% | 0.4818 | 0.4817 |
| Tuned RBF SVM | 48.28% | 0.4738 | 0.4719 |
| Neural Network (64 hidden neurons) | **53.20%** | **0.51** | **0.52** |

### Neural Network Experiments

The hidden-layer size was varied while keeping the other training settings fixed.

| Hidden Neurons | Test Accuracy |
|---:|---:|
| 12 | 48.6% |
| 32 | 50.0% |
| 64 | 53.2% |
| 128 | 48.3% |

The 64-neuron configuration achieved the highest test accuracy among the evaluated neural-network configurations.

### Discussion

The neural network achieved the highest test accuracy among the evaluated models, exceeding the polynomial SVM by 4.64 percentage points. The polynomial SVM was the strongest classical model by test accuracy and macro F1.

The results also show that performance varies across genre classes. In the neural-network evaluation, heavy-metal, hip-hop, and techno achieved F1 scores of approximately 0.71, while blues achieved 0.21.

These findings apply to the selected dataset, ten audio features, and twelve genre-based classes. Because genre labels are used as proxies for playlist classes, the experiment does not directly demonstrate personalized playlist recommendation.

## 6. Implementation

The project is implemented in Python.

Main libraries:

- pandas
- NumPy
- scikit-learn
- PyTorch
- matplotlib
- joblib

### Project Structure

```text
playlist-classifier/
│
├── artifacts/
│   ├── classes.json
│   ├── nn.pt
│   └── scaler.joblib
│
├── data/
│   ├── README.md
│   └── dataset.csv
│
├── docs/
│   ├── CHECKLIST.md
│   ├── COMMIT_PLAN.md
│   ├── PRESENTATION_OUTLINE.md
│   ├── TEAM_CONTRIBUTIONS.md
│   ├── VIVA_QA.md
│   ├── WORK_SPLIT.md
│   └── WRITEUP.md
│
├── results/
│   ├── nn_confusion.png
│   ├── nn_curves.png
│   ├── nn_experiments.txt
│   └── nn_report.txt
│
├── src/
│   ├── common.py
│   ├── data_prep.py
│   ├── demo.py
│   ├── train_baselines.py
│   └── train_nn.py
│
├── README.md
└── requirements.txt