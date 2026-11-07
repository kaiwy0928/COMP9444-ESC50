"""Build a single self-contained archive that can be copied to the cloud box.

Only the artefacts that are expensive to recreate are included: the two cached
feature arrays and the PANNs checkpoint. The raw ESC-50 wav files are not
needed because preprocessing has already been done.

Usage:
    python scripts/make_cloud_bundle.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "COMP9444_cloud_bundle.zip"

# Directories copied in full (python sources and the helper scripts).
DIRS = ["src", "models", "scripts", "cloud"]

# Individual files that stay at the archive root.
ROOTS = ["run_model.py", "1_get_raw_data.py", "2_preprocess.py",
         "3_train_models.py", "4_make_results.py",
         "requirements.txt", "requirements_cloud.txt"]

# Large binary artefacts, copied as-is.
DATA = [
    "data/processed/logmel/logmel_sr22050_m128_fft1024_hop512.npz",
    "data/processed/logmel/logmel_sr22050_m128_fft1024_hop512.json",
    "data/processed/delta/logmel_sr22050_m128_fft1024_hop512_delta9.npz",
    "data/processed/delta/logmel_sr22050_m128_fft1024_hop512_delta9.json",
    "models/pretrained/Cnn14_mAP=0.431.pth",
]


def add(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    if path.is_file():
        # Feature arrays and the checkpoint are already compressed, so level 1
        # keeps the build fast without meaningfully changing the size.
        zf.write(path, arcname, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)


def main() -> None:
    files: list[tuple[Path, str]] = []

    for rel in ROOTS:
        p = ROOT / rel
        if p.exists():
            files.append((p, rel))
        else:
            print(f"skip (missing): {rel}")

    for d in DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".sh", ".md"}:
                files.append((p, str(p.relative_to(ROOT))))

    for rel in DATA:
        p = ROOT / rel
        if p.exists():
            files.append((p, rel))
        else:
            print(f"MISSING ARTEFACT: {rel}")

    # The cloud start guide doubles as the hand-off note for the other machine.
    guide = ROOT / "cloud" / "CLOUD_START.md"
    if guide.exists():
        files.append((guide, "CLOUD_START.md"))

    total = sum(p.stat().st_size for p, _ in files)
    print(f"{len(files)} files, {total / 1e6:.0f} MB uncompressed")

    with zipfile.ZipFile(OUT, "w", allowZip64=True) as zf:
        for p, arc in files:
            add(zf, p, arc)

    print(f"written {OUT}  ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
