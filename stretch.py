"""Cut the stretch of a track a clip is made around, and make it long enough."""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

# One audio latent of LTX-2 (25 per second). `a2v` counts the latents a clip needs and refuses a track that yields
# fewer: measured at 145 frames (6.04 s), 5.98 s is refused and 6.00 s passes, so the margin is what keeps a track
# that ends with the clip from sitting on that edge. What runs past the clip is never heard: `a2v` trims it on the mux.
LATENT_SECONDS = 0.04


def on_the_grid(side: int) -> int:
    """The nearest size `ltx-2-mlx` takes. The first pass runs at half size on a latent of 32 px, so sides move in 64s.

    A caller asks for round numbers: 720p is 1280x720, and 720 is off this grid.
    """
    return max(64, round(side / 64) * 64)


def edit_size(clip_width: int, clip_height: int, short_side: int) -> tuple[int, int]:
    """The clip's aspect with its shorter side at `short_side`, on the grid: a vertical clip stays vertical."""
    scale = short_side / min(clip_width, clip_height)
    return on_the_grid(round(clip_width * scale)), on_the_grid(round(clip_height * scale))


def edit_frames(seconds: float, fps: float, clip_frames: int) -> int:
    """How long an edit is, on the 8n+1 grid: what was asked, never more than the clip has.

    The CUDA level's own expression, so both edit the same span: the guide is the edited frame and then the clip's
    frames up to one short of this, so a clip of 8n frames still fills 8n+1.
    """
    return min(math.floor(seconds * fps / 8) * 8 + 1, math.floor(clip_frames / 8) * 8 + 1)


def frames_for(seconds: float, fps: float) -> int:
    """The shortest length on LTX's 8n+1 grid that covers `seconds`. A clip is never shorter than what was asked."""
    return max(1, math.ceil((seconds * fps - 1) / 8)) * 8 + 1


def fit_stretch(waveform: np.ndarray, sample_rate: int, start: float, frames: int, fps: float) -> np.ndarray:
    """`(channels, samples)` from `start`, covering the clip, with silence where the track runs out."""
    if waveform.ndim != 2:
        raise ValueError(f"Expected (channels, samples), got shape {waveform.shape}.")
    needed = math.ceil((frames / fps + LATENT_SECONDS) * sample_rate)
    first = round(start * sample_rate)
    if first >= waveform.shape[1]:
        raise ValueError(
            f"The clip starts at {start:.2f} s and the audio lasts {waveform.shape[1] / sample_rate:.2f} s."
        )
    stretch = waveform[:, first : first + needed]
    missing = needed - stretch.shape[1]
    if missing <= 0:
        return stretch
    return np.concatenate([stretch, np.zeros((waveform.shape[0], missing), dtype=waveform.dtype)], axis=1)


def write_wav(path: Path, stretch: np.ndarray, sample_rate: int) -> None:
    """16-bit PCM: `wave` is the standard library, and ComfyUI's environment may have nothing else that writes one."""
    pcm = (np.clip(stretch, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as file:
        file.setnchannels(stretch.shape[0])
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(pcm.T.tobytes())
