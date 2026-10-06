# Playlist Classification using Spotify Audio Features
<Name 1 (SRN)>, <Name 2 (SRN)> | UE24CS352A Machine Learning Mini-Project | Problem <N>

(Export to PDF when done. Ask faculty whether it must be 1 or 2 pages; keep fonts at 10-11pt.)

## 1. Problem Statement
Playlist curation is slow, and collaborative-filtering recommendations ignore an individual's taste.
Given a song's audio features, we classify it into the most suitable playlist among C playlists
(multi-class classification), following Awadelkarim & Coelho (2018).

## 2. Dataset
- Source: Kaggle "Spotify Tracks Dataset" (<N> tracks, 114 genres).
- Used: <C> genres as playlists (<list>), up to <150> tracks each = <total> tracks.
- Features (10): danceability, energy, key, loudness, mode, speechiness, acousticness,
  liveness, valence, tempo.
- Cleaning: removed tracks listed under multiple genres (ambiguous labels) and missing values.
- Split: 80% train / 20% test, stratified, fixed seed.

## 3. Approach
- Preprocessing: standardization (zero mean, unit variance) fit on training data only.
  Features have very different scales (loudness in dB, tempo in BPM, others in [0,1]).
- Models: Perceptron; SVM with RBF, sigmoid, polynomial kernels (5-fold CV, tuned C and gamma);
  neural network with 1 hidden layer (sigmoid), LogSoftmax output, NLL loss, Adam, L2 regularization.
- Metric: test accuracy, plus per-class precision, recall, F1.

## 4. Implementation Overview
Python; scikit-learn for Perceptron/SVM; PyTorch for the neural network; pandas/matplotlib for data
and plots. `common.py` handles loading and preprocessing; `train_baselines.py` and `train_nn.py`
train and evaluate; `demo.py` predicts the playlist for a given song. Code: <GitHub URL>.

## 5. Results
| Model | Test accuracy |
|---|---|
| Perceptron | |
| RBF SVM (tuned) | |
| Sigmoid SVM | |
| Poly SVM | |
| Neural network | |

(Insert nn_curves.png and the confusion matrix of the best model.)

## 6. Conclusions
- Best model: <...> with <..>% test accuracy.
- Scaling features improved accuracy from <..> to <..> (e.g., RBF SVM).
- Easy classes: <...>; hard classes: <...> because <overlapping sound / similar genres>.
- Limitations: only audio features, genre as proxy for playlist, small dataset.
- Future work: genre/artist embeddings (word2vec/node2vec), per-user real playlists, decision trees.
