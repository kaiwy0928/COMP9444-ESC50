"""Parallel range downloader for large wheels (single-connection mirrors are slow).

Resumes an existing partial file: everything before its current size is kept and the
remaining bytes are fetched by N threads, each writing its own byte range.

    python scripts/fast_download.py <url> <out_path> [threads]
"""
import os
import sys
import threading
import time
import urllib.request

CHUNK = 4 * 1024 * 1024


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, path: str, bar: dict, lock: threading.Lock):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "r+b") as f:
        pos = start
        f.seek(pos)
        while pos <= end:
            data = r.read(min(CHUNK, end - pos + 1))
            if not data:
                break
            f.write(data)
            pos += len(data)
            with lock:
                bar["done"] += len(data)


def main():
    url, path = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    total = content_length(url)
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have > total:
        have = 0
    if have == total:
        print(f"already complete ({total / 1e9:.2f} GB)")
        return

    # preallocate so the range writes land at the right offsets
    with open(path, "r+b" if os.path.exists(path) else "wb") as f:
        if have == 0:
            f.truncate(total)

    remaining = total - have
    step = remaining // threads
    ranges = []
    for i in range(threads):
        s = have + i * step
        e = total - 1 if i == threads - 1 else s + step - 1
        ranges.append((s, e))

    print(f"total {total / 1e9:.2f} GB, have {have / 1e9:.2f} GB, "
          f"downloading {remaining / 1e9:.2f} GB with {threads} threads")
    bar = {"done": 0}
    lock = threading.Lock()
    t0 = time.time()

    def worker(r):
        fetch_range(url, r[0], r[1], path, bar, lock)

    ts = [threading.Thread(target=worker, args=(r,), daemon=True) for r in ranges]
    for t in ts:
        t.start()
    while any(t.is_alive() for t in ts):
        time.sleep(2)
        el = time.time() - t0
        pct = 100 * bar["done"] / remaining
        print(f"\r  {pct:5.1f}%  {bar['done'] / 1e6:7.1f} MB  "
              f"{bar['done'] / 1e6 / max(el, 1e-6):6.2f} MB/s", end="", flush=True)
    for t in ts:
        t.join()
    print()

    size = os.path.getsize(path)
    print(f"done: {size / 1e9:.2f} GB in {time.time() - t0:.0f}s"
          + ("  SIZE MISMATCH" if size != total else ""))


if __name__ == "__main__":
    main()
