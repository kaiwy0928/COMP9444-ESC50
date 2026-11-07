"""Download the official ESC-50 dataset into data/raw/ESC-50-master/ (about 600 MB; skipped if it is already there).

The archive is the fixed commit 33c8ce9 of https://github.com/karolpiczak/ESC-50 (the master branch since 2023-10-13),
so everyone gets the same files; scripts/verify_all.py then checks every file against that commit.
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ESC50  # noqa: E402

COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
URL = f"https://github.com/karolpiczak/ESC-50/archive/{COMMIT}.zip"


def main():
    if (ESC50 / "meta" / "esc50.csv").exists() and len(list((ESC50 / "audio").glob("*.wav"))) == 2000:
        print(f"dataset already in {ESC50}: download skipped")
        return
    raw = ESC50.parent
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / f"ESC-50-{COMMIT[:7]}.zip"
    print(f"downloading {URL}\n  -> {zip_path}")
    with urllib.request.urlopen(URL, timeout=60) as r, open(zip_path, "wb") as f:
        done, last = 0, 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if done - last >= 50 << 20:
                print(f"  {done >> 20} MB", flush=True)
                last = done
    print(f"  {done >> 20} MB downloaded; unzipping")
    tmp = raw / "_unzip"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    shutil.rmtree(ESC50, ignore_errors=True)
    (tmp / f"ESC-50-{COMMIT}").rename(ESC50)
    tmp.rmdir()
    zip_path.unlink()
    print(f"dataset ready in {ESC50}")


if __name__ == "__main__":
    main()
