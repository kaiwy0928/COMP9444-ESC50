"""Fingerprints of the preprocessed arrays, to compare a fresh run with ours clip by clip.

    python scripts/fingerprint.py save      write baseline/expected_results/preprocessing_fingerprint.npz (done once)
    python scripts/fingerprint.py compare   compare data/processed/ with it (step 2 runs this)

Saved per clip: the SHA-256 of its log-mel and delta arrays, plus mean / std / min / max, so a difference can be
traced to the exact clip and its size seen.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DELTA_DIR, PROJECT, FeatureConfig, delta_name, load_cache  # noqa: E402

FILE = PROJECT / "baseline" / "expected_results" / "preprocessing_fingerprint.npz"


def arrays() -> dict:
    mel = load_cache(FeatureConfig())
    with np.load(DELTA_DIR / f"{delta_name()}.npz", allow_pickle=True) as d:
        D = d["D"]
    return {"filename": mel["filename"], "y": mel["y"], "fold": mel["fold"], "logmel": mel["X"], "delta": D}


def per_clip(a: np.ndarray) -> dict:
    return {"sha256": np.array([hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in a]),
            "mean": a.mean(axis=(1, 2)), "std": a.std(axis=(1, 2)), "min": a.min(axis=(1, 2)), "max": a.max(axis=(1, 2))}


def save():
    a = arrays()
    out = {"filename": a["filename"], "y": a["y"], "fold": a["fold"]}
    for name in ("logmel", "delta"):
        for k, v in per_clip(a[name]).items():
            out[f"{name}_{k}"] = v
    np.savez_compressed(FILE, **out)
    print(f"saved {FILE}")


def compare() -> bool:
    if not FILE.exists():
        print(f"no {FILE.name}: nothing to compare with")
        return True
    exp = np.load(FILE, allow_pickle=True)
    a = arrays()
    ok = True
    same = (exp["filename"] == a["filename"]).all() and (exp["y"] == a["y"]).all() and (exp["fold"] == a["fold"]).all()
    print(f"clip order, labels and folds identical to ours: {same}")
    ok &= bool(same)
    for name, unit in (("logmel", "dB"), ("delta", "dB/frame")):
        mine = per_clip(a[name])
        identical = mine["sha256"] == exp[f"{name}_sha256"]
        print(f"\n{name}: {identical.sum()} / {len(identical)} clips bit-identical to ours")
        if identical.all():
            continue
        ok = False
        dmean = np.abs(mine["mean"] - exp[f"{name}_mean"])
        dmax = np.abs(mine["max"] - exp[f"{name}_max"])
        bad = np.flatnonzero(~identical)
        print(f"  {len(bad)} clips differ; largest difference in clip mean {dmean.max():.2e} {unit}, "
              f"in clip max {dmax.max():.2e} {unit}")
        for i in bad[np.argsort(-dmean[bad])][:10]:
            print(f"  {a['filename'][i]}: mean {mine['mean'][i]:.4f} vs ours {exp[name + '_mean'][i]:.4f}, "
                  f"std {mine['std'][i]:.4f} vs {exp[name + '_std'][i]:.4f}")
        print("  Tiny differences (around 1e-5) usually come from a different CPU or library build; large ones point to "
              "a different audio file or library version (check requirements.txt).")
    print("\nPREPROCESSING IDENTICAL TO OURS" if ok else "\nPREPROCESSING DIFFERS FROM OURS (details above)")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["save"]:
        save()
    else:
        compare()
