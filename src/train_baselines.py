"""Perceptron + SVM (RBF / sigmoid / poly) baselines, scaled vs unscaled."""
import argparse
import joblib
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from common import load_data, save_meta, save_confusion, ART, RES


def main(a):
    d = load_data(per_class=a.per_class)
    save_meta(d)
    models = {
        "Perceptron": lambda: Perceptron(random_state=42, max_iter=1000),
        "RBF SVM": lambda: SVC(kernel="rbf"),
        "Sigmoid SVM": lambda: SVC(kernel="sigmoid"),
        "Poly SVM": lambda: SVC(kernel="poly"),
    }
    rows = []
    for name, make in models.items():
        row = {"model": name}
        for label, X in [("scaled", d.Xtr), ("unscaled", d.Xtr_raw)]:
            s = cross_val_score(make(), X, d.ytr, cv=5)       # 5-fold CV like the paper
            row[f"cv_{label}"] = f"{s.mean():.3f} +/- {s.std():.3f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    print("\n5-fold CV accuracy (train split):\n", table.to_string(index=False))
    table.to_csv(RES / "baseline_cv.csv", index=False)

    # tune the RBF SVM (C and gamma) on scaled data
    grid = GridSearchCV(SVC(kernel="rbf"),
                        {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]},
                        cv=5, n_jobs=-1).fit(d.Xtr, d.ytr)
    print("\nBest RBF params:", grid.best_params_)
    pred = grid.predict(d.Xte)
    acc = accuracy_score(d.yte, pred)
    print(f"Tuned RBF SVM TEST accuracy: {acc:.3f}\n")
    rep = classification_report(d.yte, pred, target_names=d.classes)
    print(rep)
    (RES / "svm_report.txt").write_text(f"Test accuracy: {acc:.3f}\n"
                                        f"Params: {grid.best_params_}\n\n{rep}")
    save_confusion(d.yte, pred, d.classes, RES / "svm_confusion.png", "Tuned RBF SVM")
    joblib.dump(grid.best_estimator_, ART / "svm.joblib")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per_class", type=int, default=150)
    main(p.parse_args())
