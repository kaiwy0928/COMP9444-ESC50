"""End-to-end check of every data file in the project. Prints PASS / FAIL per check and exits non-zero on any FAIL.

 A. Download    every file in data/raw/ESC-50-master is byte-identical to the official GitHub repo
                (git blob SHA-1 vs the repo tree of karolpiczak/ESC-50 at commit 33c8ce9; needs internet, skipped with --offline)
 B. Audio       the checks in the repo's own tests/test_dataset.py: 2000 files, 44.1 kHz, mono, 220500 samples,
                both signs present, |DC offset| < 0.2
 C. Metadata    filename {fold}-{clip}-{take}-{target}.wav agrees with esc50.csv; class ids <-> names; the 50 classes and
                5 major categories match the project brief (Figure 2); 5 stratified folds
 D. Folds       our train / val / test split uses the official folds, the three parts never overlap,
                and every clip is tested exactly once
 E. Processed   log-mel .npz rows are in csv order with the right labels and folds; random clips recomputed from audio
 F. Augmented   waveaug .npz is in the same order, has 4 distinct copies per clip, and each copy still resembles
                its original
 E2. CWT        the CWT arrays: same order / labels / folds, random clips recomputed, images in the right folders
 E3. Delta      the delta channel: same clip order, equals the derivative of the stored log-mel, images in place
 G. Images      every PNG sits in the folder of its own class (class id in the filename = folder id)
 H. Organised   raw/by_class and raw/by_fold hold each wav once, in the right class / fold, identical to audio/
 I. Results     every saved predictions.npz (baseline/, model_results/, results/) has the official fold and label for
                each file, each clip once

Usage: python scripts/verify_all.py [--offline]
"""
import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import (AUG_DIR, CWT_DIR, DELTA_DIR, ESC50, LOGMEL_DIR, PROCESSED, PROJECT, CWTConfig,  # noqa: E402
                      FeatureConfig, cache_path, class_names, cwt_db, fold_split, load_audio, load_cache, load_meta,
                      log_mel)

FAILS = []
COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"  # the ESC-50 version scripts/download_data.py fetches

# The 50 classes of the project brief (Figure 2), grouped by major category, written as esc50.csv names them
BRIEF = {
    "Animals": ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow"],
    "Natural soundscapes & water sounds": ["rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
                                           "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm"],
    "Human, non-speech sounds": ["crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps",
                                 "laughing", "brushing_teeth", "snoring", "drinking_sipping"],
    "Interior/domestic sounds": ["door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks",
                                 "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick",
                                 "glass_breaking"],
    "Exterior/urban noises": ["helicopter", "chainsaw", "siren", "car_horn", "engine", "train", "church_bells",
                              "airplane", "fireworks", "hand_saw"],
}


def check(cond: bool, msg: str):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def a_download():
    print("\n== A. Download matches the official GitHub repo")
    api = "https://api.github.com/repos/karolpiczak/ESC-50"
    try:
        head = json.load(urllib.request.urlopen(f"{api}/commits/{COMMIT}", timeout=30))
        tree = json.load(urllib.request.urlopen(f"{api}/git/trees/{COMMIT}?recursive=1", timeout=60))
    except Exception as e:  # no internet, proxy, GitHub rate limit ...
        print(f"SKIP  could not reach GitHub ({type(e).__name__}); the other checks still run")
        return
    check(not tree.get("truncated"), f"repo tree complete (commit {head['sha'][:10]}, {head['commit']['committer']['date']})")
    remote = {t["path"]: t["sha"] for t in tree["tree"] if t["type"] == "blob"}
    local = {p.relative_to(ESC50).as_posix(): p for p in ESC50.rglob("*") if p.is_file()}
    check(set(remote) == set(local), f"same file list: {len(remote)} files in the repo, {len(local)} local "
                                     f"(missing {sorted(set(remote) - set(local))[:3]}, extra {sorted(set(local) - set(remote))[:3]})")
    bad = [p for p, sha in remote.items() if p in local and git_blob_sha1(local[p]) != sha]
    check(not bad, f"every file byte-identical to the repo (git SHA-1): {len(bad)} differ {bad[:3]}")
    wav = sum(p.startswith("audio/") for p in remote)
    print(f"      {wav} audio files + meta/esc50.csv + README verified against {api}")


def b_audio(meta):
    print("\n== B. Audio (the repo's own tests/test_dataset.py checks)")
    import soundfile as sf
    files = sorted((ESC50 / "audio").glob("*.wav"))
    check(len(files) == 2000, f"2000 recordings in audio/: {len(files)}")
    problems = []
    for f in files:
        x, sr = sf.read(f, dtype="float32", always_2d=True)
        if not (sr == 44100 and x.shape == (220500, 1) and x.max() > 0 and x.min() < 0 and abs(x.mean()) < 0.2):
            problems.append(f.name)
    check(not problems, f"44.1 kHz, mono, 220500 samples (5 s), both signs present, |DC| < 0.2: {len(problems)} fail {problems[:3]}")
    check(sorted(f.name for f in files) == sorted(meta["filename"]), "audio/ and esc50.csv list the same 2000 files")


def c_metadata(meta):
    print("\n== C. Metadata")
    pat = re.compile(r"^(\d)-(\d+)-([A-Z])-(\d+)\.wav$")
    bad = []
    for r in meta.itertuples():
        m = pat.match(r.filename)
        if not m or (int(m[1]), int(m[2]), m[3], int(m[4])) != (r.fold, int(r.src_file), r.take, r.target):
            bad.append(r.filename)
    check(not bad, f"filename fold / clip id / take / target agree with the csv columns: {len(bad)} disagree {bad[:3]}")
    pairs = meta[["target", "category"]].drop_duplicates()
    check(len(pairs) == 50 and pairs["target"].nunique() == 50 and pairs["category"].nunique() == 50,
          "each class id has exactly one name and vice versa")
    names = dict(zip(pairs["category"], pairs["target"]))
    brief = {c for cs in BRIEF.values() for c in cs}
    check(brief == set(names), f"the 50 classes match the project brief (missing {brief - set(names)}, "
                               f"extra {set(names) - brief})")
    check(all(names[c] // 10 == i for i, cs in enumerate(BRIEF.values()) for c in cs),
          "class id // 10 gives the major category in the brief's order")
    check(meta.groupby("target").size().eq(40).all(), "40 clips per class")
    check(meta.groupby(["fold", "target"]).size().eq(8).all() and meta["fold"].nunique() == 5,
          "5 folds, 8 clips of every class in every fold (the official prearranged folds)")


def d_folds(meta):
    print("\n== D. Our cross-validation splits use the official folds")
    fold = meta["fold"].to_numpy()
    tested = np.zeros(len(meta), int)
    for k in range(1, 6):
        train, val, test = fold_split(k)
        tr, va, te = np.isin(fold, train), fold == val, fold == k
        ok = not (tr & va).any() and not (tr & te).any() and not (va & te).any() and (tr | va | te).all()
        check(ok and tr.sum() == 1200 and va.sum() == 400 and te.sum() == 400,
              f"test fold {k}: train folds {train} (1200), val fold {val} (400), test fold {k} (400), no overlap")
        tested += te
    check((tested == 1).all(), "every clip is in the test set exactly once over the 5 runs")
    src = meta.groupby("src_file")["fold"].nunique()
    print(f"      note: {int((src > 1).sum())} of {len(src)} source recordings span two folds in the official csv "
          "(different classes from one recording); kept as published")


def e_processed(meta):
    print("\n== E. Processed log-mel arrays")
    cfg = FeatureConfig()
    d = load_cache(cfg)
    check(d["X"].shape == (2000, cfg.n_mels, 216) and d["X"].dtype == np.float32, f"X shape {d['X'].shape} float32")
    check((d["filename"] == meta["filename"].to_numpy()).all(), "rows are in esc50.csv order")
    check((d["y"] == meta["target"].to_numpy()).all(), "y equals the csv target for every row")
    check((d["fold"] == meta["fold"].to_numpy()).all(), "fold equals the csv fold for every row")
    check(np.isfinite(d["X"]).all(), "no NaN / inf")
    rng = np.random.default_rng(0)
    idx = rng.choice(2000, 12, replace=False)
    diffs = [np.abs(log_mel(load_audio(ESC50 / "audio" / d["filename"][i], cfg.sample_rate), cfg) - d["X"][i]).max()
             for i in idx]
    check(max(diffs) < 1e-3, f"12 random clips recomputed from audio match the stored arrays (max diff {max(diffs):.2e} dB)")
    rows = (PROCESSED / "clips.csv").read_text(encoding="utf-8").splitlines()[1:]
    check([r.split(",")[1] for r in rows] == list(meta["filename"]), "clips.csv lists the files in the same order as X")
    return d


def f_augmented(meta, d):
    print("\n== F. Augmented copies")
    from src.augment import AugConfig
    path = AUG_DIR / f"{FeatureConfig().name}_{AugConfig().name}.npz"
    if not path.exists():
        print("SKIP  no waveaug file")
        return
    with np.load(path, allow_pickle=True) as a:
        Xa, fn = a["X_aug"], a["filename"]
    check(Xa.shape == (2000, AugConfig().copies, 128, 216), f"X_aug shape {Xa.shape}")
    check((fn == meta["filename"].to_numpy()).all(), "same clip order as the original arrays")
    X = d["X"]
    same = sum(np.allclose(Xa[i, c].astype(np.float32), X[i], atol=0.05) for i in range(2000) for c in range(Xa.shape[1]))
    check(same == 0, f"every copy differs from its original ({same} identical)")
    # a copy should look more like its own original than like a random other clip of a different class
    rng = np.random.default_rng(1)
    idx = rng.choice(2000, 200, replace=False)
    own, other = [], []
    for i in idx:
        j = rng.choice(np.flatnonzero(d["y"] != d["y"][i]))
        a0 = Xa[i, 0].astype(np.float32).mean(1)
        own.append(np.corrcoef(a0, X[i].mean(1))[0, 1])
        other.append(np.corrcoef(a0, X[j].mean(1))[0, 1])
    check(np.median(own) > np.median(other),
          f"copies stay close to their own clip (median spectral correlation {np.median(own):.2f} vs {np.median(other):.2f} "
          "for another class)")


def e2_cwt(meta):
    print("\n== E2. CWT scalograms")
    cfg = CWTConfig()
    if not cache_path(cfg).exists():
        print("SKIP  no CWT file")
        return
    d = load_cache(cfg)
    check(d["X"].shape == (2000, cfg.n_scales, 216), f"X shape {d['X'].shape}")
    check((d["filename"] == meta["filename"].to_numpy()).all() and (d["y"] == meta["target"].to_numpy()).all()
          and (d["fold"] == meta["fold"].to_numpy()).all(), "rows in esc50.csv order with the csv target and fold")
    check(np.isfinite(d["X"]).all(), "no NaN / inf")
    idx = np.random.default_rng(2).choice(2000, 8, replace=False)
    diff = max(np.abs(cwt_db(load_audio(ESC50 / "audio" / d["filename"][i], cfg.sample_rate), cfg) - d["X"][i]).max()
               for i in idx)
    check(diff < 1e-2, f"8 random clips recomputed from audio match (max diff {diff:.2e} dB)")
    g_images(meta, CWT_DIR, "CWT")


def e3_delta(meta):
    print("\n== E3. Delta channel")
    from src.data import DELTA_WIDTH, delta_name
    path = DELTA_DIR / f"{delta_name()}.npz"
    if not path.exists():
        print("SKIP  no delta file")
        return
    import librosa
    with np.load(path, allow_pickle=True) as d:
        D, fn = d["D"], d["filename"]
    X = load_cache(FeatureConfig())["X"]
    check(D.shape == X.shape and (fn == meta["filename"].to_numpy()).all(), f"D {D.shape}, same clip order as log-mel")
    idx = np.random.default_rng(3).choice(2000, 20, replace=False)
    diff = np.abs(librosa.feature.delta(X[idx], width=DELTA_WIDTH, axis=-1, mode="interp") - D[idx]).max()
    check(diff < 1e-4, f"equals the width-{DELTA_WIDTH} time derivative of the stored log-mel (max diff {diff:.1e})")
    g_images(meta, DELTA_DIR, "delta")


def g_images(meta, root=LOGMEL_DIR, label="log-mel"):
    print(f"\n== G. {label} images ({root.relative_to(PROJECT).as_posix()})")
    names = class_names()
    pngs = list((root / "images").glob("*/*.png"))
    check(len(pngs) == 2000, f"2000 PNGs: {len(pngs)}")
    wrong = [p for p in pngs if int(p.stem.rsplit("-", 1)[1]) != int(p.parent.name[:2])
             or p.parent.name != f"{int(p.parent.name[:2]):02d}_{names[int(p.parent.name[:2])]}"]
    check(not wrong, f"every image is in the folder of its own class: {len(wrong)} misplaced {[str(p) for p in wrong[:2]]}")
    check(sorted(p.stem + ".wav" for p in pngs) == sorted(meta["filename"]), "one image per clip, names match the wavs")
    check(len(list((root / "overview").glob("*.png"))) == 50, "50 class overview sheets")


def h_organised(meta):
    print("\n== H. Raw audio organised by class / fold")
    raw = ESC50.parent
    if not (raw / "by_class").exists():
        print("SKIP  data/raw/by_class not built (optional: python scripts/organize_raw.py)")
        return
    names = class_names()
    for folder, depth in (("by_class", 1), ("by_fold", 2)):
        files = list((raw / folder).glob("*/" * depth + "*.wav"))
        check(len(files) == 2000 and len({f.name for f in files}) == 2000, f"{folder}: 2000 distinct wavs")
        r = meta.set_index("filename")
        bad = []
        for f in files:
            t, k = int(r.at[f.name, "target"]), int(r.at[f.name, "fold"])
            if f.parent.name != f"{t:02d}_{names[t]}" or (depth == 2 and f.parent.parent.name != f"fold{k}") \
                    or f.stat().st_size != (ESC50 / "audio" / f.name).stat().st_size:
                bad.append(str(f.relative_to(raw)))
        check(not bad, f"{folder}: every wav in its csv class{' and fold' if depth == 2 else ''}, same size as audio/: "
                       f"{len(bad)} wrong {bad[:2]}")
    sample = list((raw / "by_class").glob("*/*.wav"))[::97]
    same = all(hashlib.md5(f.read_bytes()).digest() == hashlib.md5((ESC50 / "audio" / f.name).read_bytes()).digest()
               for f in sample)
    check(same, f"content identical to audio/ ({len(sample)} sampled files hashed)")


def i_results(meta):
    print("\n== I. Saved predictions")
    r = meta.set_index("filename")
    runs = sorted(p for d in ("baseline", "model_results", "results") for p in (PROJECT / d).rglob("predictions.npz"))
    if not runs:
        print("SKIP  no results yet")
    for p in runs:
        with np.load(p, allow_pickle=True) as d:
            fn, y, fold = d["filename"], d["y_true"], d["fold"]
        ok = (len(set(fn)) == len(fn) and (r.loc[fn, "target"].to_numpy() == y).all()
              and (r.loc[fn, "fold"].to_numpy() == fold).all())
        check(ok, f"{p.parent.relative_to(PROJECT).as_posix()}: {len(fn)} test predictions, each clip once, official label and fold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--raw-only", action="store_true", help="only A-D: the raw dataset and the fold splits")
    a = ap.parse_args()
    meta = load_meta()
    if not a.offline:
        a_download()
    b_audio(meta)
    c_metadata(meta)
    d_folds(meta)
    if a.raw_only:
        print(f"\n{'RAW DATA CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED'}")
        sys.exit(1 if FAILS else 0)
    d = e_processed(meta)
    f_augmented(meta, d)
    g_images(meta)
    e2_cwt(meta)
    e3_delta(meta)
    h_organised(meta)
    i_results(meta)
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
