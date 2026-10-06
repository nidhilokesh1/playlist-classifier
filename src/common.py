"""Shared data loading + preprocessing for all scripts."""
from pathlib import Path
from types import SimpleNamespace
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dataset.csv"
ART = ROOT / "artifacts"
RES = ROOT / "results"
ART.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

FEATURES = ["danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "liveness", "valence", "tempo"]
LABEL_COL = "track_genre"   # each genre acts as one "playlist"
DEFAULT_GENRES = ["classical", "jazz", "heavy-metal", "hip-hop", "country",
                  "techno", "reggae", "blues", "gospel", "opera", "punk", "ambient"]


def load_data(genres=None, per_class=150, test_size=0.2, seed=42):
    """Load CSV -> pick classes -> balance -> stratified 80/20 split -> scale."""
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found at {DATA_PATH}. See data/README.md")
    df = pd.read_csv(DATA_PATH)

    # songs listed under several genres have ambiguous labels -> drop them
    if "track_id" in df.columns:
        df = df.drop_duplicates("track_id", keep=False)

    genres = genres or DEFAULT_GENRES
    available = set(df[LABEL_COL].unique())
    present = [g for g in genres if g in available]
    if len(present) < len(genres):
        print("Warning: genres not in dataset:", sorted(set(genres) - set(present)))
    df = df[df[LABEL_COL].isin(present)].dropna(subset=FEATURES)

    parts = [g.sample(min(len(g), per_class), random_state=seed)
             for _, g in df.groupby(LABEL_COL)]
    df = pd.concat(parts).reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    X = df[FEATURES].values.astype(float)
    idx = np.arange(len(df))
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, idx, test_size=test_size, stratify=y, random_state=seed)

    scaler = StandardScaler().fit(Xtr)          # fit on TRAIN only (no leakage)
    return SimpleNamespace(
        Xtr_raw=Xtr, Xte_raw=Xte, Xtr=scaler.transform(Xtr), Xte=scaler.transform(Xte),
        ytr=ytr, yte=yte, classes=list(le.classes_), scaler=scaler,
        df_test=df.iloc[ite].reset_index(drop=True))


def save_meta(data):
    joblib.dump(data.scaler, ART / "scaler.joblib")
    (ART / "classes.json").write_text(json.dumps(data.classes))


def save_confusion(y_true, y_pred, classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=classes, xticks_rotation=45,
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
