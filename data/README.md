# Dataset

Download the **Spotify Tracks Dataset** from Kaggle
(search "Spotify Tracks Dataset", by maharshipandya; file `dataset.csv`) and save it as:

    data/dataset.csv

Required columns: `track_id`, `track_name`, `artists`, `track_genre` and the 10 audio features
(danceability, energy, key, loudness, mode, speechiness, acousticness, liveness, valence, tempo).

The CSV is git-ignored (large file). Each genre is treated as one "playlist" class.
