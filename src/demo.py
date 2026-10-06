"""Live demo: predict which playlist (genre) a song belongs to.

  python src/demo.py --random          # random song from the held-out test set
  python src/demo.py --features 0.6 0.8 5 -5.0 1 0.05 0.1 0.2 0.7 120
      (order: danceability energy key loudness mode speechiness
              acousticness liveness valence tempo)
"""
import argparse
import json
import joblib
import numpy as np
import torch
from common import ART, FEATURES, load_data
from train_nn import build


def main(a):
    classes = json.loads((ART / "classes.json").read_text())
    scaler = joblib.load(ART / "scaler.joblib")
    ck = torch.load(ART / "nn.pt")
    model = build(ck["n_in"], ck["hidden"], ck["n_out"])
    model.load_state_dict(ck["state"])
    model.eval()

    truth = None
    if a.features:
        x = np.array(a.features, dtype=float).reshape(1, -1)
    else:
        d = load_data()
        i = np.random.randint(len(d.Xte_raw))
        x = d.Xte_raw[i:i + 1]
        row = d.df_test.iloc[i]
        truth = row["track_genre"]
        print("Song:", row.get("track_name", "?"), "-", row.get("artists", "?"))
    print(dict(zip(FEATURES, np.round(x[0], 3))))

    with torch.no_grad():
        p = model(torch.tensor(scaler.transform(x), dtype=torch.float32)).exp()[0].numpy()
    print("\nTop 3 predicted playlists:")
    for j in p.argsort()[::-1][:3]:
        print(f"  {classes[j]:<12} {p[j]*100:5.1f}%")
    if truth:
        print("Actual playlist:", truth)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--features", type=float, nargs=10)
    a = ap.parse_args()
    if not a.random and not a.features:
        ap.error("use --random or --features")
    main(a)
