"""Arrange the raw ESC-50 wav files by class and by fold, without touching the official audio/ folder.

    data/raw/by_class/<id>_<class>/<file>.wav          50 folders x 40 files
    data/raw/by_fold/fold<k>/<id>_<class>/<file>.wav    5 folds x 50 classes x 8 files

Files are hard links to audio/<file>.wav (no extra disk space on NTFS); if linking fails they are copied.
Labels and folds come from meta/esc50.csv, the official metadata.

Usage: python scripts/organize_raw.py
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ESC50, load_meta  # noqa: E402

RAW = ESC50.parent


def place(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    meta = load_meta()
    for folder in ("by_class", "by_fold"):
        shutil.rmtree(RAW / folder, ignore_errors=True)
    how = {"link": 0, "copy": 0}
    for r in meta.itertuples():
        src = ESC50 / "audio" / r.filename
        cls = f"{r.target:02d}_{r.category}"
        how[place(src, RAW / "by_class" / cls / r.filename)] += 1
        how[place(src, RAW / "by_fold" / f"fold{r.fold}" / cls / r.filename)] += 1
    print(f"by_class: {len(list((RAW / 'by_class').glob('*/*.wav')))} files in "
          f"{len(list((RAW / 'by_class').iterdir()))} folders; "
          f"by_fold: {len(list((RAW / 'by_fold').glob('*/*/*.wav')))} files ({how['link']} links, {how['copy']} copies)")


if __name__ == "__main__":
    main()
