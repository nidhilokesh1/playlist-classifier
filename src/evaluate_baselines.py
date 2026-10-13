"""Evaluate classical ML models on the shared held-out test set."""

import pandas as pd
import joblib

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

from common import load_data, save_confusion, ART, RES


def main():
    d = load_data(per_class=150)
    RES.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    models = {
        "Perceptron": Perceptron(random_state=42, max_iter=1000),
        "RBF SVM": SVC(kernel="rbf"),
        "Sigmoid SVM": SVC(kernel="sigmoid"),
        "Polynomial SVM": SVC(kernel="poly"),
    }

    rows = []

    for name, model in models.items():
        model.fit(d.Xtr, d.ytr)
        pred = model.predict(d.Xte)

        acc = accuracy_score(d.yte, pred)
        macro = f1_score(d.yte, pred, average="macro", zero_division=0)
        weighted = f1_score(
            d.yte, pred, average="weighted", zero_division=0
        )

        rows.append({
            "model": name,
            "test_accuracy": acc,
            "macro_f1": macro,
            "weighted_f1": weighted,
        })

        report = classification_report(
            d.yte, pred, labels=list(range(len(d.classes))),
            target_names=d.classes, zero_division=0
        )

        safe_name = name.lower().replace(" ", "_")
        (RES / f"{safe_name}_test_report.txt").write_text(
            f"Model: {name}\n"
            f"Test accuracy: {acc:.4f}\n"
            f"Macro F1: {macro:.4f}\n"
            f"Weighted F1: {weighted:.4f}\n\n"
            f"{report}"
        )

        save_confusion(
            d.yte, pred, d.classes,
            RES / f"{safe_name}_test_confusion.png",
            f"{name} - Test Set"
        )

    # Tune RBF SVM using training data only.
    grid = GridSearchCV(
        SVC(kernel="rbf"),
        {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", 0.01, 0.1, 1],
        },
        cv=5,
        n_jobs=-1,
    )
    grid.fit(d.Xtr, d.ytr)

    pred = grid.predict(d.Xte)
    acc = accuracy_score(d.yte, pred)
    macro = f1_score(d.yte, pred, average="macro", zero_division=0)
    weighted = f1_score(
        d.yte, pred, average="weighted", zero_division=0
    )

    rows.append({
        "model": "Tuned RBF SVM",
        "test_accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
    })

    report = classification_report(
        d.yte, pred, labels=list(range(len(d.classes))),
        target_names=d.classes, zero_division=0
    )

    (RES / "tuned_rbf_svm_test_report.txt").write_text(
        f"Model: Tuned RBF SVM\n"
        f"Best parameters: {grid.best_params_}\n"
        f"Test accuracy: {acc:.4f}\n"
        f"Macro F1: {macro:.4f}\n"
        f"Weighted F1: {weighted:.4f}\n\n"
        f"{report}"
    )

    save_confusion(
        d.yte, pred, d.classes,
        RES / "tuned_rbf_svm_test_confusion.png",
        "Tuned RBF SVM - Test Set"
    )
    joblib.dump(grid.best_estimator_, ART / "svm.joblib")

    table = pd.DataFrame(rows).sort_values(
        "test_accuracy", ascending=False
    )
    table.to_csv(RES / "baseline_test_results.csv", index=False)

    print("\nHeld-out test results (same split for every model):")
    print(table.to_string(index=False, formatters={
        "test_accuracy": "{:.4f}".format,
        "macro_f1": "{:.4f}".format,
        "weighted_f1": "{:.4f}".format,
    }))
    print(f"\nBest tuned RBF parameters: {grid.best_params_}")
    print(f"Results saved in: {RES.resolve()}")


if __name__ == "__main__":
    main()
