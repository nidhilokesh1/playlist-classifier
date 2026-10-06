# Viva / Q&A Preparation
1. **What is the problem?** Multi-class classification: song features -> playlist (genre).
2. **Why scale features?** Loudness (dB) and tempo (BPM) dominate distance/kernel computations
   otherwise. Unscaled RBF SVM accuracy collapses. Scaler is fit on train data only to avoid leakage.
3. **How does an SVM work?** Finds the maximum-margin boundary. Kernels (RBF) allow non-linear
   boundaries. C trades margin width against training errors; gamma sets RBF width.
4. **How is multi-class handled?** scikit-learn SVC uses one-vs-one internally; Perceptron uses one-vs-rest.
5. **Perceptron limits?** Only linear boundaries, so it fails when classes overlap non-linearly.
6. **Neural network architecture?** 10 -> hidden (sigmoid) -> C with LogSoftmax; NLL loss is
   cross-entropy; trained by gradient descent via backpropagation.
7. **Overfitting?** High train accuracy but low test accuracy. Mitigated by L2 regularization,
   cross-validation, a small network.
8. **Why is accuracy not enough?** Class-wise precision/recall/F1 show which playlists fail.
9. **Why 80/20 and stratified?** Keeps class proportions equal in both sets.
10. **Which classes were confused and why?** Answer from your confusion matrix.
11. **How would you improve it?** More features (artist/genre embeddings), real user playlists,
    more data, other algorithms (decision trees, random forests).
12. **Dataset caveat?** Genre is a proxy for a playlist; real playlists are subjective and overlapping.
