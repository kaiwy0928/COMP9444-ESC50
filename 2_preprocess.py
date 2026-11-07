"""Step 2: raw audio -> model inputs (about 3 minutes), then check the results and compare them with ours.

data/processed/logmel/  log-mel spectrograms (128 bands x 216 frames per clip) + every clip as an image
data/processed/delta/   their first temporal derivative (the second input channel) + images
data/processed/clips.csv  one row per clip, same order as the arrays

The comparison checks every clip's arrays against our fingerprint (baseline/expected_results/
preprocessing_fingerprint.npz) and names any clip that differs, so a problem is caught here, before training.
Every experiment reads these arrays, so steps 1 and 2 are run once per machine.
"""
from src.runner import environment_report, run

if __name__ == "__main__":
    environment_report()
    run("scripts/preprocess.py")
    run("scripts/preprocess_delta.py")
    run("scripts/verify_all.py", "--offline")
    run("scripts/fingerprint.py", "compare")
