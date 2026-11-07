"""Step 1: get the raw ESC-50 dataset and check it (download about 600 MB the first time, then about 1 minute).

- downloads the official dataset (fixed commit 33c8ce9) into data/raw/ESC-50-master/ if it is not there yet
- byte-identical to the official GitHub repository (skipped without internet)
- 2000 wav files, 44.1 kHz, mono, 5 s; labels and file names agree; 50 classes as in the project brief
- the official 5 folds, and our train / validation / test splits built from them
"""
from src.runner import environment_report, run

if __name__ == "__main__":
    environment_report()
    run("scripts/download_data.py")
    run("scripts/verify_all.py", "--raw-only")
