"""Put every model in model_results/ into one table (nothing is trained):

    python scripts/summarize_models.py

For each model: parameters; development protocol (validation and test accuracy, per seed run and averaged); final
protocol (test accuracy mean +- SD over seeds, macro F1). Written to the console and model_results/summary.csv.
"""
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "model_results"


def runs(model_dir: Path, protocol: str) -> list[dict]:
    out = []
    for run in sorted(model_dir.glob(f"{protocol}_seed*")):
        if not (run / "metrics.json").exists():
            continue
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["summary"]
        c = json.loads((run / "config.json").read_text(encoding="utf-8"))
        h = json.loads((run / "history.json").read_text(encoding="utf-8"))
        val = [f["best_val_acc"] for f in h.values() if f.get("best_val_acc") is not None]
        out.append({"acc": m["accuracy"]["mean"] * 100, "f1": m["macro_f1"]["mean"] * 100,
                    "val": st.mean(val) * 100 if val else None, "params": c.get("n_params")})
    return out


def fmt(values: list[float]) -> str:
    if not values:
        return "-"
    return f"{st.mean(values):.2f}" + (f" ± {st.stdev(values):.2f}" if len(values) > 1 else "") + f" (n={len(values)})"


def main():
    rows = []
    for model_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()) if RESULTS.exists() else []:
        dev, final = runs(model_dir, "dev"), runs(model_dir, "final")
        if not dev and not final:
            continue
        params = next((r["params"] for r in dev + final if r["params"]), None)
        rows.append({"model": model_dir.name, "params_M": f"{params / 1e6:.2f}" if params else "-",
                     "dev_val_acc": fmt([r["val"] for r in dev if r["val"] is not None]),
                     "dev_test_acc": fmt([r["acc"] for r in dev]),
                     "final_test_acc": fmt([r["acc"] for r in final]),
                     "final_macro_f1": fmt([r["f1"] for r in final])})
    if not rows:
        print("no results yet in model_results/ - run run_model.py first")
        return
    keys = list(rows[0])
    widths = {k: max(len(k), *(len(r[k]) for r in rows)) for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        print("  ".join(r[k].ljust(widths[k]) for k in keys))
    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nsaved {RESULTS / 'summary.csv'}")


if __name__ == "__main__":
    main()
