"""Helpers for the numbered step files. Each step runs a project script with the same
Python interpreter, from the repository root, so it works from an IDE "Run" button or a terminal alike."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    """Run `python <args>` in ROOT; stop with a clear message if it fails."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, *args]
    print(f"\n>>> {' '.join(args)}", flush=True)
    t0 = time.time()
    code = subprocess.call(cmd, cwd=ROOT, env=env)
    if code != 0:
        sys.exit(f"\nFAILED: `python {' '.join(args)}` exited with code {code}. See the messages above.")
    print(f"<<< done in {time.time() - t0:.0f} s", flush=True)


def environment_report() -> None:
    import platform
    print(f"Python {platform.python_version()} at {sys.executable}")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        print(f"PyTorch {torch.__version__}; " + (f"GPU: {gpu}" if gpu else "no CUDA GPU found: training will run "
                                                    "on the CPU (much slower, results may differ slightly)"))
    except ImportError:
        sys.exit("PyTorch is not installed: see 'Set up the environment' in README.md.")
    if sys.version_info < (3, 11):
        sys.exit("Python 3.11 or newer is needed (numpy 2.4 / pandas 3.1).")
