"""The ComfyUI nodes: a clip from text, from a picture or around a given sound, a clip edited from its first frame, and a
clip that follows a reference, by `ltx-2-mlx` in an environment of its own."""
from __future__ import annotations

import gc
import os
import subprocess
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
MODEL = "ltx-2.5-mlx-q8"
WEIGHT = "transformer-distilled.safetensors"
# WepeNerd/LTX-Ripple's own workflow: its strength, and its prompt, which the person's description is appended to.
RIPPLE_STRENGTH = 1.35
RIPPLE_PROMPT = (
    "Use the reference video for motion, timing, camera movement, composition, and unchanged scene content, while "
    "consistently propagating the visual edit established in the first frame throughout the video."
)
# The CUDA level puts the first frame in at 0.7 for its first pass and 1.0 for its second. There is one pass here, and
# 0.7 is the one measured on a Mac (#716): the clip opened on the picture and then followed the reference's camera.
REFERENCE_FIRST_FRAME_STRENGTH = 0.7


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
    """It exited 0, so now look at the file: LTX-2.5's clips come with sound, and one without it is not the clip asked for."""
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("ltx-2-mlx finished and left no file.")
    with av.open(str(path)) as container:
        if not container.streams.video or not container.streams.audio:
            raise RuntimeError("ltx-2-mlx left a file without both video and audio.")


def start() -> tuple[Path, Path, Path]:
    """What every clip needs before it runs: the memory back from ComfyUI, the machine's yes, and the program."""
    free_memory()
    refused = memory_guard.stop_reason_at_start(memory_guard.read_machine())
    if refused:
        raise supervise.MemoryStop(f"Not started, to protect this Mac: {refused}. Close other programs and free some disk.")
    model = model_dir()
    python = executable()
    return model, python, prepare_ffmpeg(python)


def files_for_this_run() -> tuple[Path, str]:
    temp = Path(folder_paths.get_temp_directory())
    temp.mkdir(parents=True, exist_ok=True)
    return temp, f"ltx2mlx-{uuid.uuid4().hex[:12]}"


def save_first_frame(image, path: Path) -> None:
    Image.fromarray((image[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)).save(path)


def run_clip(argv: list[str], python: Path, tools: Path, output: Path, log: Path, inputs: list[Path]) -> tuple:
    """Runs the program under the guard, and whatever happens leaves nothing behind but the clip it made."""
    env = supervise.child_environment(dict(os.environ), [VENV / "bin", tools])
    bar = comfy.utils.ProgressBar(100)
    watch = memory_guard.Watch(memory_guard.read_machine, time.monotonic)

    def attempt() -> supervise.Run:
        return supervise.run_once(
            argv, str(python), env, log,
            should_stop=model_management.processing_interrupted,
            on_line=supervise.Progress(lambda percent: bar.update_absolute(percent, 100)),
            memory=watch.check,
        )

    try:
        supervise.run_with_one_retry(attempt, say=say, before_retry=free_memory)
    except supervise.Cancelled:
        say("Cancelled, and the child is gone.")
        model_management.throw_exception_if_processing_interrupted()
        raise
    finally:
        say(watch.peak_line())
        for path in inputs:
            path.unlink(missing_ok=True)

    check_output(output)
    log.unlink(missing_ok=True)
    bar.update_absolute(100, 100)
    return (InputImpl.VideoFromFile(str(output)),)


def clip_inputs() -> dict:
    """The controls every clip has."""
    return {
        "prompt": ("STRING", {"multiline": True, "default": ""}),
        "duration": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 20.0, "step": 0.01}),
        "width": ("INT", {"default": 768, "min": 64, "max": 1920, "step": 64}),
        "height": ("INT", {"default": 512, "min": 64, "max": 1920, "step": 64}),
        "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
        "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True}),
    }


def generate_clip(prompt, duration, width, height, fps, seed, image=None) -> tuple:
    """`ltx-2-mlx generate`: from text, or from a picture when there is one."""
    model, python, tools = start()
    temp, run = files_for_this_run()
    output = temp / f"{run}.mp4"
    log = temp / f"{run}.log"
    width, height = stretch.on_the_grid(width), stretch.on_the_grid(height)
    frames = stretch.frames_for(duration, fps)
    argv = [
        *supervise.program(str(python), "generate"),
        *supervise.clip_arguments(str(model), prompt, frames, width, height, fps, seed, str(output)),
    ]
    inputs = []
    if image is not None:
        first_frame = temp / f"{run}.png"
        save_first_frame(image, first_frame)
        inputs.append(first_frame)
        argv += ["--image", str(first_frame), "0", "1.0"]
    say(f"{frames} frames at {width}x{height}, {'from a picture' if inputs else 'from text'}. Log: {log}")
    return run_clip(argv, python, tools, output, log, inputs)


class LTX2MLXTextToVideo:
    """A prompt in, a clip with its sound out."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": clip_inputs()}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "10sai/LTX-2.5 MLX"

    def generate(self, prompt, duration, width, height, fps, seed):
        return generate_clip(prompt, duration, width, height, fps, seed)


class LTX2MLXImageToVideo:
    """A picture and a prompt in, a clip with its sound out that starts on that picture."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), **clip_inputs()}}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "10sai/LTX-2.5 MLX"

    def generate(self, image, prompt, duration, width, height, fps, seed):
        return generate_clip(prompt, duration, width, height, fps, seed, image=image)


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
        model, python, tools = start()
        temp, run = files_for_this_run()
        first_frame = temp / f"{run}.png"
        track = temp / f"{run}.wav"
        output = temp / f"{run}.mp4"
        log = temp / f"{run}.log"

        save_first_frame(image, first_frame)

        frames = stretch.frames_for(duration, fps)
        waveform = audio["waveform"][0].cpu().numpy()
        stretch.write_wav(track, stretch.fit_stretch(waveform, audio["sample_rate"], audio_start, frames, fps), audio["sample_rate"])

        argv = [
            *supervise.program(str(python), "a2v"),
            *supervise.clip_arguments(str(model), prompt, frames, width, height, fps, seed, str(output)),
            "--audio", str(track),
            "--image", str(first_frame),
        ]
        say(f"{frames} frames at {width}x{height}, {audio_start:.2f} s into the audio. Log: {log}")
        return run_clip(argv, python, tools, output, log, [first_frame, track])


def fitted(picture: Image.Image, width: int, height: int) -> Image.Image:
    """ComfyUI's `ImageScale` with `lanczos` and `center`, as the CUDA level scales: cut to the aspect, then resize."""
    aspect = width / height
    w, h = picture.size
    if w / h > aspect:
        cut = round(h * aspect)
        picture = picture.crop(((w - cut) // 2, 0, (w - cut) // 2 + cut, h))
    elif w / h < aspect:
        cut = round(w / aspect)
        picture = picture.crop((0, (h - cut) // 2, w, (h - cut) // 2 + cut))
    return picture.resize((width, height), Image.Resampling.LANCZOS)


def clip_frames(video, count: int):
    """The clip's first `count` frames from where its trim starts, decoded one at a time rather than all at once."""
    start, _ = video.get_active_trim_window()
    with av.open(video.get_stream_source()) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.time is not None and frame.time < start:
                continue
            yield frame.to_image()
            count -= 1
            if count == 0:
                return


def write_guide(first_frame, video, frames: int, width: int, height: int, fps: float, path: Path, tools: Path) -> None:
    """The edited frame, then the clip's frames 0 to `frames - 2`, losslessly: the guide `ic-lora` encodes."""
    first = Image.fromarray((first_frame[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8))
    encoder = subprocess.Popen(
        [str(tools / "ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path)],
        stdin=subprocess.PIPE,
    )
    written = 0
    try:
        for picture in (first, *clip_frames(video, frames - 1)):
            encoder.stdin.write(fitted(picture.convert("RGB"), width, height).tobytes())
            written += 1
    finally:
        encoder.stdin.close()
        encoder.wait()
    if encoder.returncode != 0 or written != frames:
        raise RuntimeError(f"The guide came out with {written} of {frames} frames.")


def without_sound(made: Path, muted: Path, tools: Path) -> None:
    """`ic-lora` invents a track; the CUDA level returns none, so the clip keeps its own wherever it is used."""
    subprocess.run([str(tools / "ffmpeg"), "-v", "error", "-y", "-i", str(made), "-an", "-c:v", "copy", str(muted)], check=True)
    made.unlink(missing_ok=True)


class LTX2MLXRippleEdit:
    """A clip and its first frame with an edit on it in; the same clip out, with the edit carried through it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_video": ("VIDEO",),
                "first_frame": ("IMAGE",),
                # Picked like `LoraLoaderModelOnly` picks it, so the canvas names the weight it loads.
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {"default": 4, "min": 1, "max": 4}),
                "height": ("INT", {"default": 512, "min": 64, "max": 1920, "step": 64, "tooltip": "The shorter side."}),
                "seed": ("INT", {"default": 132644, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "edit"
    CATEGORY = "10sai/LTX-2.5 MLX"

    def edit(self, reference_video, first_frame, lora_name, prompt, duration, height, seed):
        lora = folder_paths.get_full_path("loras", lora_name)
        if not lora:
            raise RuntimeError(f"{lora_name} is not installed: install the level again from the companion.")
        model, python, tools = start()
        temp, run = files_for_this_run()
        guide = temp / f"{run}-guide.mp4"
        made = temp / f"{run}-made.mp4"
        output = temp / f"{run}.mp4"
        log = temp / f"{run}.log"

        fps = float(reference_video.get_frame_rate())
        width, height = stretch.edit_size(*reference_video.get_dimensions(), height)
        frames = stretch.edit_frames(duration, fps, reference_video.get_frame_count())
        write_guide(first_frame, reference_video, frames, width, height, fps, guide, tools)

        described = f"{RIPPLE_PROMPT} {prompt.strip()}".strip()
        argv = [
            *supervise.program(str(python), "ic-lora"),
            *supervise.clip_arguments(str(model), described, frames, width, height, fps, seed, str(made)),
            "--single-stage",
            "--lora", lora, str(RIPPLE_STRENGTH),
            "--video-conditioning", str(guide), "1.0",
        ]
        say(f"{frames} frames at {width}x{height}, {fps:g} fps, edited from the first frame. Log: {log}")
        run_clip(argv, python, tools, made, log, [guide])
        without_sound(made, output, tools)
        return (InputImpl.VideoFromFile(str(output)),)


def write_frames(frames, count: int, fps: float, path: Path, tools: Path) -> tuple[int, int]:
    """The first `count` frames of an IMAGE batch, losslessly and one at a time, as they come: `ic-lora` sizes the guide."""
    _, height, width, _ = frames.shape
    encoder = subprocess.Popen(
        [str(tools / "ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path)],
        stdin=subprocess.PIPE,
    )
    try:
        for index in range(count):
            encoder.stdin.write((frames[index, :, :, :3].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8).tobytes())
    finally:
        encoder.stdin.close()
        encoder.wait()
    if encoder.returncode != 0:
        raise RuntimeError(f"The guide could not be written: ffmpeg exited {encoder.returncode}.")
    return width, height


class LTX2MLXReferenceVideo:
    """A guide (a previz's depth, or a clip that already is one) in; a clip with its framing, camera and motion out."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Frames rather than a VIDEO, because the guide is made in the canvas: MoGe's depth of the clip, or the clip itself.
                "guide": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {"default": 4, "min": 1, "max": 4}),
                "height": ("INT", {"default": 512, "min": 64, "max": 1920, "step": 64, "tooltip": "The shorter side."}),
                # The name the companion reads as the level's guide strength.
                "reference_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "10sai/LTX-2.5 MLX"

    def generate(self, guide, fps, lora_name, prompt, duration, height, reference_strength, seed, first_frame=None):
        lora = folder_paths.get_full_path("loras", lora_name)
        if not lora:
            raise RuntimeError(f"{lora_name} is not installed: install the level again from the companion.")
        model, python, tools = start()
        temp, run = files_for_this_run()
        guide_file = temp / f"{run}-guide.mp4"
        output = temp / f"{run}.mp4"
        log = temp / f"{run}.log"

        frames = stretch.reference_frames(duration, fps, guide.shape[0])
        guide_width, guide_height = write_frames(guide, frames, fps, guide_file, tools)
        width, height = stretch.edit_size(guide_width, guide_height, height)
        inputs = [guide_file]
        picture = None
        if first_frame is not None:
            picture = temp / f"{run}.png"
            fitted(Image.fromarray((first_frame[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)), width, height).save(picture)
            inputs.append(picture)

        argv = [
            *supervise.program(str(python), "ic-lora"),
            *supervise.clip_arguments(str(model), prompt.strip(), frames, width, height, round(fps), seed, str(output)),
            *supervise.reference_arguments(
                lora, str(guide_file), reference_strength, str(picture) if picture else None, REFERENCE_FIRST_FRAME_STRENGTH
            ),
        ]
        say(f"{frames} frames at {width}x{height}, {fps:g} fps, following a guide at {reference_strength:g}"
            f"{', from a first frame' if picture else ''}. Log: {log}")
        return run_clip(argv, python, tools, output, log, inputs)


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


NODE_CLASS_MAPPINGS = {
    "LTX2MLXTextToVideo": LTX2MLXTextToVideo,
    "LTX2MLXImageToVideo": LTX2MLXImageToVideo,
    "LTX2MLXAudioToVideo": LTX2MLXAudioToVideo,
    "LTX2MLXRippleEdit": LTX2MLXRippleEdit,
    "LTX2MLXReferenceVideo": LTX2MLXReferenceVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2MLXTextToVideo": "LTX-2.5 text to video (MLX)",
    "LTX2MLXImageToVideo": "LTX-2.5 image to video (MLX)",
    "LTX2MLXAudioToVideo": "LTX-2.5 audio to video (MLX)",
    "LTX2MLXRippleEdit": "LTX-2.5 Ripple edit from the first frame (MLX)",
    "LTX2MLXReferenceVideo": "LTX-2.5 video that follows a reference (MLX)",
}
