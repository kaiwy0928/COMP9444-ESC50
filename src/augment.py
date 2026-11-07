"""Offline waveform augmentation (after Salamon & Bello, 2017 [2]).

For every clip we make N augmented copies. Each copy applies a random chain, with at least one step, of:
    pitch shift   ±1 or ±2 semitones            ([2]: the most helpful augmentation, never hurt any class)
    time stretch  rate in [0.85, 1.15], then cropped / zero-padded back to 5 s
    noise         white / pink / brown noise at 10–30 dB SNR
                  ([2] mixed in real street / park scenes; ESC-50 has no such background set, so we use coloured noise)
Every copy is peak-normalised again and turned into a log-mel spectrogram with the same parameters as the original.
Only training folds ever use the copies; validation and test always see the original clips.
Copies are seeded per clip, so a rebuild gives identical arrays.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import AUG_DIR, ESC50, FeatureConfig, load_audio, load_meta, log_mel


@dataclass(frozen=True)
class AugConfig:
    copies: int = 4
    p_pitch: float = 0.5
    pitch_steps: tuple = (-2, -1, 1, 2)
    p_stretch: float = 0.5
    stretch_range: tuple = (0.85, 1.15)
    p_noise: float = 0.5
    snr_db_range: tuple = (10.0, 30.0)
    seed: int = 2026

    @property
    def name(self) -> str:
        return f"waveaug{self.copies}"


def coloured_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """white: flat spectrum; pink: power ~ 1/f; brown: power ~ 1/f^2."""
    white = rng.standard_normal(n)
    if kind == "white":
        return white
    spec = np.fft.rfft(white)
    f = np.arange(len(spec), dtype=float)
    f[0] = 1.0
    spec /= np.sqrt(f) if kind == "pink" else f
    return np.fft.irfft(spec, n)


def fix_length(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(y) >= n:
        start = rng.integers(0, len(y) - n + 1)
        return y[start:start + n]
    return np.pad(y, (0, n - len(y)))


def augment_once(y: np.ndarray, sr: int, cfg: AugConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    import librosa
    n = len(y)
    steps = []
    while not steps:  # at least one augmentation per copy
        steps = [s for s, p in (("pitch", cfg.p_pitch), ("stretch", cfg.p_stretch), ("noise", cfg.p_noise))
                 if rng.random() < p]
    info = {}
    if "pitch" in steps:
        info["pitch"] = int(rng.choice(cfg.pitch_steps))
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=info["pitch"])
    if "stretch" in steps:
        info["stretch"] = round(float(rng.uniform(*cfg.stretch_range)), 3)
        y = fix_length(librosa.effects.time_stretch(y, rate=info["stretch"]), n, rng)
    if "noise" in steps:
        kind = str(rng.choice(["white", "pink", "brown"]))
        snr = float(rng.uniform(*cfg.snr_db_range))
        noise = coloured_noise(n, kind, rng)
        sig_power = np.mean(y ** 2) + 1e-12
        noise *= np.sqrt(sig_power / (10 ** (snr / 10)) / (np.mean(noise ** 2) + 1e-12))
        y = y + noise
        info["noise"] = f"{kind}@{snr:.1f}dB"
    peak = np.abs(y).max()
    return (y / peak if peak > 0 else y).astype(np.float32), info


def _clip_copies(i: int, filename: str, feat: FeatureConfig, cfg: AugConfig):
    rng = np.random.default_rng([cfg.seed, i])
    y = load_audio(ESC50 / "audio" / filename, feat.sample_rate)
    specs, infos = [], []
    for _ in range(cfg.copies):
        ya, info = augment_once(y, feat.sample_rate, cfg, rng)
        specs.append(log_mel(ya, feat).astype(np.float16))
        infos.append(info)
    return np.stack(specs), infos


def build_aug_cache(feat: FeatureConfig = FeatureConfig(), cfg: AugConfig = AugConfig(), n_jobs: int = 8) -> Path:
    """-> data/processed/augmented/<feat.name>_<cfg.name>.npz with X_aug (2000, copies, n_mels, frames) float16 in clip order."""
    from joblib import Parallel, delayed
    from tqdm import tqdm
    meta = load_meta()
    out = Parallel(n_jobs=n_jobs)(delayed(_clip_copies)(i, fn, feat, cfg)
                                  for i, fn in enumerate(tqdm(meta["filename"], desc="augment")))
    X_aug = np.stack([o[0] for o in out])
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    path = AUG_DIR / f"{feat.name}_{cfg.name}.npz"
    np.savez_compressed(path, X_aug=X_aug, filename=meta["filename"].to_numpy())
    (AUG_DIR / f"{feat.name}_{cfg.name}.json").write_text(json.dumps(
        {"features": asdict(feat), "augmentation": asdict(cfg),
         "per_clip": {fn: o[1] for fn, o in zip(meta["filename"], out)}}, indent=1))
    return path


def load_aug_cache(feat: FeatureConfig = FeatureConfig(), cfg: AugConfig = AugConfig()) -> np.ndarray:
    path = AUG_DIR / f"{feat.name}_{cfg.name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python scripts/augment_offline.py` first")
    with np.load(path, allow_pickle=True) as d:
        return d["X_aug"]
