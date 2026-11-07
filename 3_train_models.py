"""Step 3: train and evaluate the five models of the report (needs steps 1 and 2).

Each model is trained with the settings written in its own file (models/model_<name>.py, the SETTINGS list):
standard ESC-50 5-fold cross-validation, seeds 42, 43 and 44, the common recipe of run_model.py. The three seeds run
at the same time (one process each, PARALLEL below), as we did; a seed gives the same result alone or in parallel.
Results are written to model_results/model_<name>/final_seed42/ ... final_seed44/, replacing the ones in the
repository. Our own results are already there, so step 4 works without running this step.

Click Run, or from a terminal:
    python 3_train_models.py                    # the models in MODELS below
    python 3_train_models.py resnet18           # only these models
    python 3_train_models.py --quick            # check that everything runs: 1 fold x 3 epochs per model, a few
                                                # minutes; results go to smoke_* folders that step 4 leaves out
On other hardware the numbers differ slightly from ours (README.md, "Reproducing our results").
"""
import importlib.util
import os
import subprocess
import sys
import time

from src.runner import ROOT, environment_report, run

MODELS = ["baseline", "multiscale_cnn", "mobilenetv3_small", "efficientnet_b0", "resnet18"]
QUICK = False   # True = the quick check described above (same as --quick)
PARALLEL = 1    # seeds trained at the same time; 1 = one after another (less GPU memory)
                # set to 3 on a large GPU; this machine has 4 GB, where two runs do not fit


def model_settings(model_file: str) -> list[str]:
    """The SETTINGS list at the end of a model file (what clicking Run on that file would use)."""
    spec = importlib.util.spec_from_file_location("settings_of_" + model_file.replace("/", "_"), ROOT / model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __name__ is not "__main__", so nothing is trained here
    return list(getattr(module, "SETTINGS", ["--seeds", "42", "43", "44"]))


def split_seeds(settings: list[str]) -> tuple[list[str], list[str]]:
    """(seeds, the other settings) from a SETTINGS list such as ["--seeds", "42", "43", "44", "--epochs", "30"]."""
    if "--seeds" not in settings:
        return ["42"], settings
    i = settings.index("--seeds")
    j = i + 1
    while j < len(settings) and not settings[j].startswith("-"):
        j += 1
    return settings[i + 1:j], settings[:i] + settings[j:]


def train_seeds_in_parallel(model_file: str, seeds: list[str], rest: list[str]):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    logs = ROOT / "results" / "step3_logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending, running, failed = list(seeds), [], []
    t0 = time.time()
    print(f"\n>>> {model_file}: seeds {' '.join(seeds)} ({min(PARALLEL, len(seeds))} at a time), settings {' '.join(rest)}",
          flush=True)
    while pending or running:
        while pending and len(running) < PARALLEL:
            seed = pending.pop(0)
            log = logs / f"{os.path.basename(model_file)[:-3]}_seed{seed}.log"
            cmd = [sys.executable, "run_model.py", model_file, "--seeds", seed, *rest]
            running.append((seed, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(log, "w", encoding="utf-8"),
                                                   stderr=subprocess.STDOUT), log))
            print(f"    seed {seed} started (progress: {log.relative_to(ROOT)})", flush=True)
        time.sleep(5)
        for item in list(running):
            seed, proc, log = item
            if proc.poll() is None:
                continue
            running.remove(item)
            tail = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if "Accuracy" in line]
            print(f"    seed {seed} {'finished' if proc.returncode == 0 else 'FAILED'}"
                  f"{': ' + tail[-1].strip() if tail else ''}", flush=True)
            if proc.returncode != 0:
                failed.append(seed)
    if failed:
        sys.exit(f"\nFAILED: {model_file} seeds {failed}; see results/step3_logs/")
    print(f"<<< {model_file} done in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    quick = QUICK or "--quick" in args
    chosen = [a for a in args if not a.startswith("--")] or MODELS
    unknown = [m for m in chosen if m not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s) {unknown}; choose from {MODELS}")
    environment_report()
    for name in chosen:
        model_file = f"models/model_{name}.py"
        if quick:
            run("run_model.py", model_file, "--folds", "1", "--epochs", "3")
        else:
            seeds, rest = split_seeds(model_settings(model_file))
            train_seeds_in_parallel(model_file, seeds, rest)
    print("\nStep 3 finished. Next: 4_make_results.py")
