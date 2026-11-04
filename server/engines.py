"""Model wrappers. All audio is float32 numpy at 48 kHz unless a function says otherwise.

Models load lazily on first use and stay in memory while files are queued, then go when the queue empties.
"""
from __future__ import annotations

import ctypes
import gc
import importlib
import os
import threading
import urllib.request
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchaudio

from . import media

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
SR = 48000

Progress = Callable[[float], None]
_noop: Progress = lambda _: None
_lock = threading.Lock()

# CPU beat the GPU for the separator on an M1 Pro (0.85x vs 0.61x realtime). Six threads held up best
# when other work was competing for the cores; more threads made it far slower.
torch.set_num_threads(6)


def _resample(x: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    return x if src == dst else torchaudio.functional.resample(x, src, dst)


class Separator:
    """BandIt v2 (multilingual): splits a mix into speech, music and effects, what the sliders control."""

    CHUNK = 384000  # 8 s, the length it was trained on
    OVERLAP = 96000  # 2 s
    URL = "https://zenodo.org/api/records/12701995/files/checkpoint-multi.ckpt/content"
    STEMS = ["speech", "music", "sfx"]
    CONFIG = dict(
        in_channels=1,
        stems=STEMS,
        band_type="musical",
        n_bands=64,
        normalize_channel_independently=False,
        treat_channel_as_feature=True,
        n_sqm_modules=8,
        emb_dim=128,
        rnn_dim=256,
        bidirectional=True,
        rnn_type="GRU",
        mlp_dim=512,
        hidden_activation="Tanh",
        hidden_activation_kwargs=None,
        complex_mask=True,
        use_freq_weights=True,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        window_fn="hann_window",
        wkwargs=None,
        power=None,
        center=True,
        normalized=True,
        pad_mode="reflect",
        onesided=True,
        fs=SR,
    )

    def __init__(self) -> None:
        self.net = None

    def _weights(self) -> Path:
        """Downloads the checkpoint once and keeps only the model weights (149 MB of its 447 MB)."""
        folder = MODELS_DIR / "bandit_v2"
        path = folder / "model.pt"
        if path.exists():
            return path
        folder.mkdir(parents=True, exist_ok=True)
        part = folder / "checkpoint.part"
        urllib.request.urlretrieve(self.URL, part)
        state = torch.load(part, map_location="cpu", weights_only=False)["state_dict"]
        torch.save({k.removeprefix("model."): v for k, v in state.items()}, path)
        part.unlink()
        return path

    def load(self) -> None:
        if self.net is not None:
            return
        from .vendor.bandit_v2.bandit import Bandit

        net = Bandit(**self.CONFIG)
        net.load_state_dict(torch.load(self._weights(), map_location="cpu", weights_only=True))
        self.net = net.float().eval()

    @torch.inference_mode()
    def __call__(self, stereo: np.ndarray, progress: Progress = _noop) -> dict[str, np.ndarray]:
        """stereo: (n, 2). Returns speech (n, 2), music (n, 2), sfx (n, 2)."""
        self.load()
        peak = float(np.abs(stereo).max()) or 1.0
        # The masks behave best near full scale; the gain is undone on the way out.
        gain = 0.9 / peak if peak < 0.9 else 1.0
        wav = torch.from_numpy(stereo.T.copy() * gain)
        length = wav.shape[1]
        ramp = torch.linspace(0, 1, self.OVERLAP)
        window = torch.ones(self.CHUNK)
        window[: self.OVERLAP] = ramp
        window[-self.OVERLAP :] = ramp.flip(0)
        out = torch.zeros(len(self.STEMS), 2, length + self.CHUNK)
        norm = torch.zeros(length + self.CHUNK)
        starts = list(range(0, max(1, length), self.CHUNK - self.OVERLAP))
        for i, start in enumerate(starts):
            seg = wav[:, start : start + self.CHUNK]
            pad = self.CHUNK - seg.shape[1]
            if pad > 0:
                seg = torch.nn.functional.pad(seg, (0, pad))
            # Shape: (batch, stem, channel, samples).
            result = self.net(seg.unsqueeze(0))[0]
            out[:, :, start : start + self.CHUNK] += result * window
            norm[start : start + self.CHUNK] += window
            progress((i + 1) / len(starts))
        out = out[:, :, :length] / norm[:length].clamp_min(1e-3) / gain
        return {name: out[s].T.numpy() for s, name in enumerate(self.STEMS)}


class SpeechDenoiser:
    """MossFormer2 48 kHz: strips residual noise and room sound. Keeps the phase of its input."""

    def __init__(self) -> None:
        self.cv = None

    def load(self) -> None:
        if self.cv is not None:
            return
        # ClearVoice reads and downloads checkpoints relative to the working directory.
        here = Path.cwd()
        target = MODELS_DIR / "clearvoice"
        target.mkdir(parents=True, exist_ok=True)
        os.chdir(target)
        try:
            # An import inside it keeps an ImportError for good, and with it every frame that led there, so a running job's files
            # would stay held after the job. A thread of its own has no such frames.
            first = threading.Thread(target=importlib.import_module, args=("clearvoice",))
            first.start()
            first.join()
            from clearvoice import ClearVoice

            cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"]).models[0]
            # When its download fails ClearVoice only prints a warning, then runs on untrained weights and reports success.
            if not (target / "checkpoints" / "MossFormer2_SE_48K" / "last_best_checkpoint").is_file():
                raise media.MediaError("The v1 model could not be downloaded. Check your connection and try again.")
            self.cv = cv
        finally:
            os.chdir(here)

    @torch.inference_mode()
    def __call__(self, mono: np.ndarray) -> np.ndarray:
        self.load()
        from clearvoice.utils.decode import decode_one_audio

        out = decode_one_audio(self.cv.model, self.cv.device, mono[None, :].astype(np.float32), self.cv.args)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        return np.pad(out[: len(mono)], (0, max(0, len(mono) - len(out))))


class FastDenoiser:
    """DeepFilterNet3. About 35x realtime; used by both chains."""

    def __init__(self) -> None:
        self.model = None

    def load(self) -> None:
        if self.model is None:
            import df.logger
            from df.enhance import init_df

            # It shells out to git for a log line, which prints a confusing error outside a repository.
            df.logger.get_commit_hash = lambda: None
            self.model, self.state, _ = init_df(log_level="ERROR")

    @torch.inference_mode()
    def __call__(self, mono: np.ndarray) -> np.ndarray:
        self.load()
        from df.enhance import enhance

        y = enhance(self.model, self.state, torch.from_numpy(mono.copy()).unsqueeze(0).float())
        return y.squeeze(0).cpu().numpy().astype(np.float32)


class SpeechRestorer:
    """Sidon: regenerates speech from cleaned features, the step that makes it sound studio-recorded."""

    REPO = "sarulab-speech/sidon-v0.1"
    CHUNK_S = 96
    # Its output runs this many samples late, measured on word onsets on both reference voices and on
    # every offset of its frame grid. Removed here so callers get audio that lines up with the input.
    DELAY = 360

    def __init__(self) -> None:
        self.fe = None

    def load(self) -> None:
        if self.fe is not None:
            return
        import transformers
        from huggingface_hub import hf_hub_download

        self.fe = torch.jit.load(hf_hub_download(self.REPO, "feature_extractor_cpu.pt"), map_location="cpu").eval()
        self.dec = torch.jit.load(hf_hub_download(self.REPO, "decoder_cpu.pt"), map_location="cpu").eval()
        self.pre = transformers.SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    @torch.inference_mode()
    def __call__(self, mono: np.ndarray, progress: Progress = _noop) -> np.ndarray:
        self.load()
        target = len(mono)
        peak = float(np.abs(mono).max())
        if peak < 1e-5:
            return np.zeros(target, np.float32)
        wav = torch.from_numpy(0.9 * mono / peak).float().view(1, -1)
        wav = torchaudio.functional.highpass_biquad(wav, SR, 50)
        wav16 = torch.nn.functional.pad(_resample(wav, SR, 16000), (0, 24000))
        chunks = wav16.view(-1).split(16000 * self.CHUNK_S)
        outs, cache = [], None
        for i, chunk in enumerate(chunks):
            inputs = self.pre(torch.nn.functional.pad(chunk, (160, 160)).numpy(), sampling_rate=16000, return_tensors="np")
            feature = self.fe(torch.from_numpy(inputs["input_features"]))["last_hidden_state"]
            if cache is not None:
                feature = torch.cat([cache, feature], dim=1)
            outs.append(self.dec(feature.transpose(1, 2)).view(-1)[:-960])
            cache = feature[:, -1:]
            progress((i + 1) / len(chunks))
        y = torch.cat(outs).numpy()[self.DELAY : target + self.DELAY].astype(np.float32)
        return np.pad(y, (0, max(0, target - len(y))))


def speech_regions(mono: np.ndarray) -> list[tuple[int, int]]:
    """Sample ranges containing speech, from the Silero voice-activity model bundled with faster-whisper."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = _resample(torch.from_numpy(mono.copy()), SR, 16000).numpy()
    options = VadOptions(threshold=0.4, min_silence_duration_ms=300, speech_pad_ms=200)
    return [(t["start"] * 3, t["end"] * 3) for t in get_speech_timestamps(audio, options)]


class Engines:
    def __init__(self) -> None:
        self.separator = Separator()
        self.denoiser = SpeechDenoiser()
        self.fast_denoiser = FastDenoiser()
        self.restorer = SpeechRestorer()


_engines: Engines | None = None


def get_engines() -> Engines:
    global _engines
    with _lock:
        if _engines is None:
            _engines = Engines()
        return _engines


def release_memory() -> None:
    """Collects freed objects and hands their pages back to the system."""
    gc.collect()
    try:
        # macOS otherwise keeps freed pages in its allocator, and the server looks just as large.
        ctypes.CDLL(None).malloc_zone_pressure_relief(None, 0)
    except (AttributeError, OSError):
        pass


def release_models() -> None:
    """Drops every loaded model, so an idle server stays small. The next job loads them again, which takes a few seconds."""
    global _engines
    with _lock:
        if _engines is None:
            return
        _engines = None
    from faster_whisper.vad import get_vad_model

    get_vad_model.cache_clear()
    release_memory()
