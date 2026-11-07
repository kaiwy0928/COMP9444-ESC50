"""Check whether a model file fits in GPU memory at a given batch size.

    python scripts/check_vram.py models/model_panns_cnn14.py            # try 32, 16, 8, 4, 2
    python scripts/check_vram.py models/model_panns_cnn14.py 16         # one batch size

Prints parameters, forward/backward peak memory and the throughput, so we know the
largest batch size that fits before committing to a long cross-validation run.
"""
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"probe_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_model


def probe(path: Path, batch: int) -> bool:
    build = load(path)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = build(in_channels=2, n_classes=50).to(device)
        x = torch.randn(batch, 2, 128, 216, device=device)
        y = torch.randint(0, 50, (batch,), device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        t0 = time.time()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 3
        peak = torch.cuda.max_memory_allocated() / 2**30
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  batch {batch:3d}: OK  params {n_par / 1e6:5.2f}M  "
              f"peak {peak:4.2f} GiB  {dt * 1000:5.0f} ms/step  "
              f"{batch / dt:6.1f} samples/s")
        del model, x, y, opt
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        print(f"  batch {batch:3d}: OUT OF MEMORY")
        torch.cuda.empty_cache()
        return False
    except Exception:
        traceback.print_exc()
        return False


def main():
    path = Path(sys.argv[1])
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    print(f"model: {path.name}")
    if len(sys.argv) > 2:
        probe(path, int(sys.argv[2]))
        return 0
    for batch in (32, 16, 8, 4, 2, 1):
        if probe(path, batch):
            print(f"  -> largest tested batch that fits: {batch}")
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
