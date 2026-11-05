"""The ComfyUI node: a clip made around a given sound, by `ltx-2-mlx` in an environment of its own."""
from __future__ import annotations

import gc
import os
import time
import uuid
from pathlib import Path

import av
import numpy as np
import folder_paths
import comfy.model_management as model_management
import comfy.utils
from comfy_api.latest import InputImpl
from PIL import Image

from . import memory_guard, stretch, supervise

PACK = Path(__file__).parent
VENV = PACK / ".venv"
TOOLS = PACK / ".bin"
FOLDER_KEY = "ltx-mlx"
MODEL = "ltx-2.5-mlx-q4"
WEIGHT = "transformer-distilled.safetensors"


def model_dir() -> Path:
    for root in folder_paths.get_folder_paths(FOLDER_KEY):
        candidate = Path(root) / MODEL
        if (candidate / WEIGHT).exists():
            return candidate
    raise RuntimeError(f"{MODEL} is not installed: install the level again from the companion.")


def executable() -> Path:
    python = supervise.python_of(VENV)
    if not python.exists():
        raise RuntimeError("The LTX-2.5 MLX environment is missing: install the level again from the companion.")
    return python


def say(message: str) -> None:
    print(f"[LTX2-MLX] {message}", flush=True)


def free_memory() -> None:
    # On a 16 GB Mac, whatever ComfyUI still holds comes straight out of what the child has.
    model_management.unload_all_models()
    model_management.soft_empty_cache(True)
    gc.collect()


def check_output(path: Path) -> None:
    """It exited 0, so now look at the file: a clip without its sound would defeat the whole point of this node."""
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("ltx-2-mlx finished and left no file.")
    with av.open(str(path)) as container:
        if not container.streams.video or not container.streams.audio:
            raise RuntimeError("ltx-2-mlx left a file without both video and audio.")


class LTX2MLXAudioToVideo:
    """Image plus audio in, clip with that same audio out, with the mouth following it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("FLOAT", {"default": 6.0, "min": 0.5, "max": 20.0, "step": 0.01}),
                "audio_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
                "width": ("INT", {"default": 448, "min": 64, "max": 1920, "step": 64}),
                "height": ("INT", {"default": 768, "min": 64, "max": 1920, "step": 64}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "10sai/LTX-2.5 MLX"

    def generate(self, image, audio, prompt, duration, audio_start, width, height, fps, seed):
        free_memory()
        refused = memory_guard.stop_reason_at_start(memory_guard.read_machine())
        if refused:
            raise supervise.MemoryStop(f"Not started, to protect this Mac: {refused}. Close other programs and free some disk.")
        model = model_dir()
        python = executable()
        tools = prepare_ffmpeg(python)

        temp = Path(folder_paths.get_temp_directory())
        temp.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:12]
        first_frame = temp / f"ltx2mlx-{run_id}.png"
        track = temp / f"ltx2mlx-{run_id}.wav"
        output = temp / f"ltx2mlx-{run_id}.mp4"
        log = temp / f"ltx2mlx-{run_id}.log"

        Image.fromarray((image[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)).save(first_frame)

        frames = stretch.frames_for(duration, fps)
        waveform = audio["waveform"][0].cpu().numpy()
        stretch.write_wav(track, stretch.fit_stretch(waveform, audio["sample_rate"], audio_start, frames, fps), audio["sample_rate"])

        argv = [
            *supervise.program(str(python)),
            "--model", str(model),
            "--prompt", prompt,
            "--audio", str(track),
            "--image", str(first_frame),
            "--height", str(height), "--width", str(width),
            "--frames", str(frames), "--frame-rate", str(fps),
            "--seed", str(seed),
            "--output", str(output),
        ]
        env = supervise.child_environment(dict(os.environ), [VENV / "bin", tools])
        bar = comfy.utils.ProgressBar(100)

        def attempt() -> supervise.Run:
            return supervise.run_once(
                argv, str(python), env, log,
                should_stop=model_management.processing_interrupted,
                on_line=supervise.Progress(lambda percent: bar.update_absolute(percent, 100)),
                memory=watch.check,
            )

        watch = memory_guard.Watch(memory_guard.read_machine, time.monotonic)
        say(f"{frames} frames at {width}x{height}, {audio_start:.2f} s into the audio. Log: {log}")
        try:
            supervise.run_with_one_retry(attempt, say=say, before_retry=free_memory)
        except supervise.Cancelled:
            say("Cancelled, and the child is gone.")
            model_management.throw_exception_if_processing_interrupted()
            raise
        finally:
            say(watch.peak_line())
            first_frame.unlink(missing_ok=True)
            track.unlink(missing_ok=True)

        check_output(output)
        log.unlink(missing_ok=True)
        bar.update_absolute(100, 100)
        return (InputImpl.VideoFromFile(str(output)),)


def prepare_ffmpeg(python: Path) -> Path:
    """`ltx-2-mlx` looks for `ffmpeg` on PATH and hard-codes libx264, which a Mac does not come with.

    The pack's environment carries it through `imageio-ffmpeg`, whose binary has a versioned name, so this puts the
    name the pipeline looks for in a folder of ours that goes first on the child's PATH.
    """
    TOOLS.mkdir(exist_ok=True)
    link = TOOLS / "ffmpeg"
    if not link.exists():
        import subprocess

        found = subprocess.run(
            [str(python), "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        link.unlink(missing_ok=True)
        link.symlink_to(found)
    return TOOLS


NODE_CLASS_MAPPINGS = {"LTX2MLXAudioToVideo": LTX2MLXAudioToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"LTX2MLXAudioToVideo": "LTX-2.5 audio to video (MLX)"}
