"""ESC-50 preprocessing and PyTorch datasets.

Pipeline (no denoising, see README):
    wav (44.1 kHz) -> resample to 22.05 kHz -> peak-normalise -> log-mel spectrogram (128 bands)   [FeatureConfig]
                                                                -> CWT scalogram (128 scales)      [CWTConfig]
    log-mel + its first temporal derivative (delta) as a second channel                             ["logmel_delta"]
Both give a (128, 216) array per clip, so the same models run on either. Features of all 2000 clips are computed once
and stored in data/processed/<representation>/, so training only reads arrays.

Splits follow the official 5 folds: for test fold k, fold (k mod 5) + 1 is the validation fold (used to pick the
best epoch) and the other three folds are for training. Normalisation statistics come from the training folds only.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT = Path(__file__).resolve().parents[1]
ESC50 = PROJECT / "data" / "raw" / "ESC-50-master"
PROCESSED = PROJECT / "data" / "processed"
LOGMEL_DIR = PROCESSED / "logmel"      # one folder per representation: arrays + images + previews
DELTA_DIR = PROCESSED / "delta"
CWT_DIR = PROCESSED / "cwt"
AUG_DIR = PROCESSED / "augmented"


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 22050
    n_fft: int = 1024       # ~46 ms window
    hop_length: int = 512   # ~23 ms hop -> 216 frames per 5 s clip
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 11025.0   # Nyquist at 22.05 kHz
    amin: float = 1e-10     # floor before log, so silent padding becomes a finite value

    @property
    def name(self) -> str:
        return f"logmel_sr{self.sample_rate}_m{self.n_mels}_fft{self.n_fft}_hop{self.hop_length}"


@dataclass(frozen=True)
class CWTConfig:
    """Continuous wavelet transform with an analytic Morlet wavelet (Torrence & Compo, 1998).

    128 centre frequencies, geometrically spaced from 40 Hz to 10.5 kHz (a CWT has no 0 Hz scale). Power |W|^2 is
    averaged over blocks of 512 samples -> 216 frames, the same time grid as the log-mel features, then converted to dB.
    The wavelet is scaled so a sinusoid of amplitude A gives |W| = A at its own frequency, whatever the scale.
    """
    sample_rate: int = 22050
    n_scales: int = 128
    fmin: float = 40.0
    fmax: float = 10500.0
    omega0: float = 6.0     # Morlet centre frequency (cycles): the usual time / frequency trade-off
    hop_length: int = 512
    amin: float = 1e-10

    @property
    def name(self) -> str:
        return (f"cwt_morlet{self.omega0:g}_sr{self.sample_rate}_k{self.n_scales}"
                f"_f{self.fmin:g}-{self.fmax:g}_hop{self.hop_length}")

    def frequencies(self) -> np.ndarray:
        return np.geomspace(self.fmin, self.fmax, self.n_scales)


FEATURES = {"logmel": FeatureConfig(), "cwt": CWTConfig()}
FEATURE_CHOICES = [*FEATURES, "logmel_delta"]
DELTA_WIDTH = 9  # librosa.feature.delta window (frames): a smoothed first derivative over ~0.2 s


def delta_name(cfg: FeatureConfig = FeatureConfig()) -> str:
    return f"{cfg.name}_delta{DELTA_WIDTH}"


def load_meta() -> pd.DataFrame:
    """esc50.csv: filename, fold, target, category, esc10, src_file, take."""
    return pd.read_csv(ESC50 / "meta" / "esc50.csv")


def class_names() -> list[str]:
    meta = load_meta().drop_duplicates("target").sort_values("target")
    return meta["category"].tolist()


MAJOR_CATEGORIES = ["Animals", "Natural soundscapes", "Human non-speech", "Interior/domestic", "Exterior/urban"]


def load_audio(path: Path, sr: int) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True, res_type="soxr_hq")
    peak = np.abs(y).max()
    return y / peak if peak > 0 else y  # peak normalisation removes recording-level loudness differences


def log_mel(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                                         n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0)
    return librosa.power_to_db(mel, ref=1.0, amin=cfg.amin, top_db=None).astype(np.float32)  # (n_mels, frames)


def cwt_db(y: np.ndarray, cfg: CWTConfig) -> np.ndarray:
    """Morlet CWT computed in the frequency domain: W_s = IFFT( FFT(y) * psi_hat(s * omega) ), psi_hat zero for
    negative frequencies (analytic wavelet). Runs on the GPU when one is available. Returns (n_scales, frames) dB."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(y)
    n_fft = 1 << int(np.ceil(np.log2(n)))          # zero-pad to a power of two (110250 -> 131072 samples)
    frames = int(np.ceil(n / cfg.hop_length))
    Y = torch.fft.fft(torch.as_tensor(y, dtype=torch.float32, device=dev), n=n_fft)
    omega = 2 * np.pi * torch.fft.fftfreq(n_fft, d=1 / cfg.sample_rate, device=dev)          # rad / s
    scales = torch.as_tensor(cfg.omega0 / (2 * np.pi * cfg.frequencies()), dtype=torch.float32, device=dev)[:, None]
    psi = 2 * torch.exp(-0.5 * (scales * omega[None] - cfg.omega0) ** 2) * (omega > 0)       # (K, n_fft)
    W = torch.fft.ifft(Y[None] * psi, dim=1)[:, :n]
    power = torch.nn.functional.pad(W.abs() ** 2, (0, frames * cfg.hop_length - n))
    power = power.reshape(cfg.n_scales, frames, cfg.hop_length).mean(-1)
    return (10 * torch.log10(power.clamp(min=cfg.amin))).cpu().numpy().astype(np.float32)


def feature_dir(cfg) -> Path:
    return CWT_DIR if isinstance(cfg, CWTConfig) else LOGMEL_DIR


def cache_path(cfg) -> Path:
    return feature_dir(cfg) / f"{cfg.name}.npz"


def compute_features(y: np.ndarray, cfg) -> np.ndarray:
    return cwt_db(y, cfg) if isinstance(cfg, CWTConfig) else log_mel(y, cfg)


def silence_ratio(y: np.ndarray, frame: int = 1024, threshold_db: float = -60.0) -> float:
    """Share of frames whose RMS is more than 60 dB below the clip peak (mostly the zero padding)."""
    n = len(y) // frame
    rms = np.sqrt((y[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    return float((20 * np.log10(rms + 1e-12) < threshold_db).mean())


def build_cache(cfg=FeatureConfig()) -> Path:
    """Compute features (log-mel or CWT) for all clips -> data/processed/<logmel|cwt>/<cfg.name>.npz."""
    from tqdm import tqdm
    meta = load_meta()
    feats, silence = [], []
    for fn in tqdm(meta["filename"], desc=cfg.name.split("_")[0]):
        y = load_audio(ESC50 / "audio" / fn, cfg.sample_rate)
        feats.append(compute_features(y, cfg))
        silence.append(silence_ratio(y))
    feature_dir(cfg).mkdir(parents=True, exist_ok=True)
    out = cache_path(cfg)
    np.savez_compressed(out, X=np.stack(feats), y=meta["target"].to_numpy(), fold=meta["fold"].to_numpy(),
                        filename=meta["filename"].to_numpy(), silence=np.array(silence, dtype=np.float32))
    (feature_dir(cfg) / f"{cfg.name}.json").write_text(json.dumps(asdict(cfg), indent=2))
    return out


def load_cache(cfg=FeatureConfig()) -> dict:
    path = cache_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python scripts/preprocess.py` "
                                f"(or preprocess_cwt.py for CWT) first")
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def build_delta(cfg: FeatureConfig = FeatureConfig()) -> Path:
    """First temporal derivative of every stored log-mel spectrogram -> data/processed/delta/<delta_name>.npz.
    Computed from the cached log-mel, which is identical to recomputing it from audio."""
    import librosa
    d = load_cache(cfg)
    D = librosa.feature.delta(d["X"], width=DELTA_WIDTH, order=1, axis=-1, mode="interp").astype(np.float32)
    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    out = DELTA_DIR / f"{delta_name(cfg)}.npz"
    np.savez_compressed(out, D=D, filename=d["filename"])
    (DELTA_DIR / f"{delta_name(cfg)}.json").write_text(json.dumps(
        {"source": cfg.name, "delta": {"width": DELTA_WIDTH, "order": 1, "axis": "time", "mode": "interp",
                                       "function": "librosa.feature.delta"}}, indent=2))
    return out


def with_delta(X_aug: np.ndarray) -> np.ndarray:
    """(N, copies, F, T) log-mel copies -> (N, copies, 2, F, T) float16 with the same delta as build_delta_cache."""
    import librosa
    D = librosa.feature.delta(X_aug.astype(np.float32), width=DELTA_WIDTH, order=1, axis=-1, mode="interp")
    return np.stack([X_aug, D.astype(np.float16)], axis=2)


def load_features(key: str) -> tuple[dict, str]:
    """(arrays, feature name) for --features: X is (N, F, T) for one channel or (N, C, F, T) for several."""
    if key != "logmel_delta":
        cfg = FEATURES[key]
        return load_cache(cfg), cfg.name
    d = load_cache(FeatureConfig())
    path = DELTA_DIR / f"{delta_name()}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python scripts/preprocess_delta.py` first")
    with np.load(path, allow_pickle=True) as dd:
        assert (dd["filename"] == d["filename"]).all(), "delta file is not in the same clip order"
        d["X"] = np.stack([d["X"], dd["D"]], axis=1)
    return d, delta_name()


def fold_split(test_fold: int) -> tuple[list[int], int, int]:
    """(training folds, validation fold, test fold) for one of the 5 cross-validation runs."""
    val_fold = test_fold % 5 + 1
    return [f for f in range(1, 6) if f not in (test_fold, val_fold)], val_fold, test_fold


class LogMelDataset(Dataset):
    """Normalised clips as (channels, n_mels, frames) tensors; X is (N, F, T) for one channel or (N, C, F, T).

    mean / std are per-channel, per-band statistics from the training folds. Training-only options (the baseline uses none):
        X_aug         (N, copies, n_mels, frames) offline waveform-augmented versions; each draw picks the original
                      or one of the copies with equal probability
        time_shift    circular shift by a random number of frames, so events can appear anywhere in the clip
        spec_augment  SpecAugment frequency / time masking
        crop          train on a random window of this many frames (sub-segment training; 0 = the whole clip)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 spec_augment: dict | None = None, X_aug: np.ndarray | None = None, time_shift: bool = False,
                 crop: int = 0):
        X = X[:, None] if X.ndim == 3 else X
        mean, std = mean.reshape(X.shape[1], -1), std.reshape(X.shape[1], -1)
        self.mean, self.std = mean[..., None].astype(np.float32), std[..., None].astype(np.float32)  # (C, F, 1)
        self.X = ((X - self.mean[None]) / self.std[None]).astype(np.float32)
        self.X_aug = X_aug  # kept in float16, normalised per draw to save memory
        self.y = y.astype(np.int64)
        self.spec_augment = spec_augment
        self.time_shift = time_shift
        self.crop = crop

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        k = np.random.randint(0, self.X_aug.shape[1] + 1) if self.X_aug is not None else 0
        if k == 0:
            x = self.X[i]
        else:  # a copy is (F, T) for one channel or (C, F, T) for several
            a = self.X_aug[i, k - 1].astype(np.float32)
            x = ((a[None] if a.ndim == 2 else a) - self.mean) / self.std
        if self.time_shift:
            x = np.roll(x, np.random.randint(0, x.shape[-1]), axis=-1)
        if self.crop:
            t0 = np.random.randint(0, x.shape[-1] - self.crop + 1)
            x = x[..., t0:t0 + self.crop]
        x = torch.from_numpy(np.ascontiguousarray(x))
        if self.spec_augment:
            x = spec_augment(x, **self.spec_augment)
        return x, self.y[i]


def norm_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean / std over clips and time: per band (F,) for (N, F, T), per channel and band (C, F) for (N, C, F, T)."""
    axes = (0, X.ndim - 1)
    return X.mean(axis=axes), X.std(axis=axes) + 1e-6


def spec_augment(x: torch.Tensor, freq_masks: int = 2, freq_width: int = 16, time_masks: int = 2,
                 time_width: int = 24) -> torch.Tensor:
    """Zero out random frequency bands and time spans (Park et al., 2019). x: (1, n_mels, frames), already normalised,
    so 0 is the mean value."""
    x = x.clone()
    _, n_mels, frames = x.shape
    for _ in range(freq_masks):
        w = np.random.randint(0, freq_width + 1)
        f0 = np.random.randint(0, max(1, n_mels - w))
        x[:, f0:f0 + w, :] = 0
    for _ in range(time_masks):
        w = np.random.randint(0, time_width + 1)
        t0 = np.random.randint(0, max(1, frames - w))
        x[:, :, t0:t0 + w] = 0
    return x
