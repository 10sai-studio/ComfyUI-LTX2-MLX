"""Runs `ltx-2-mlx` as a child: retry, cancel and what to say when it fails. Nothing here imports ComfyUI."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

GUARD = Path(__file__).with_name("guard.py")
STREAMED_PROGRAM = Path(__file__).with_name("stream_text_encoder.py")
GPU_TIMEOUT = "GPU Timeout"
STOP_GRACE_SECONDS = 3.0
POLL_SECONDS = 0.2
GUARD_EVERY_SECONDS = 2.0
TAIL_LINES = 200


class Cancelled(Exception):
    """Somebody asked to stop, and the child is gone."""


class GenerationFailed(RuntimeError):
    pass


class MemoryStop(GenerationFailed):
    """The memory guard stopped it. Never retried: the same run would meet the same machine."""


def program(python: str) -> list[str]:
    """`-P` because running a script puts its folder first on the path, and this folder holds `nodes.py` and `tests`."""
    return [python, "-P", str(STREAMED_PROGRAM), "a2v", "--distilled", "--low-ram"]


@dataclass
class Run:
    code: int
    tail: list[str]

    @property
    def timed_out_on_gpu(self) -> bool:
        return any(GPU_TIMEOUT in line for line in self.tail)


def signal_group(pgid: int, sent: int) -> bool:
    """`False` when the group is gone. macOS answers EPERM rather than ESRCH for a group with only zombies left."""
    try:
        os.killpg(pgid, sent)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def kill_group(process: subprocess.Popen, grace: float = STOP_GRACE_SECONDS) -> None:
    """The whole group: the guard, `ltx-2-mlx` and the ffmpeg it pipes frames into."""
    for sent in (signal.SIGTERM, signal.SIGKILL):
        if not signal_group(process.pid, sent):
            break
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and signal_group(process.pid, 0):
            process.poll()
            time.sleep(0.05)
    process.wait()


def run_once(
    argv: Sequence[str],
    python: str,
    env: dict[str, str],
    log: Path,
    should_stop: Callable[[], bool],
    on_line: Callable[[str], None] = lambda _line: None,
    memory: Callable[[], Optional[str]] = lambda: None,
) -> Run:
    """One attempt. `Cancelled` when `should_stop` said so, `MemoryStop` when `memory` did, and only once the child has
    really gone. Whatever ends it, nothing of the group outlives this: `kill` on just the guard would orphan the rest."""
    process = subprocess.Popen(
        [python, str(GUARD), *argv],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    tail: deque[str] = deque(maxlen=TAIL_LINES)

    def read() -> None:
        # tqdm rewrites its line with `\r`, so a line is whatever comes between either end.
        pending = b""
        with log.open("ab") as file:
            while chunk := process.stdout.read1(8192):
                file.write(chunk)
                file.flush()
                pending += chunk
                *lines, pending = re.split(rb"[\r\n]", pending)
                for raw in lines:
                    line = raw.decode("utf-8", "replace").strip()
                    if line:
                        tail.append(line)
                        on_line(line)
            if pending.strip():
                tail.append(pending.decode("utf-8", "replace").strip())

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    next_look = time.monotonic()
    try:
        while process.poll() is None:
            if should_stop():
                raise Cancelled()
            if time.monotonic() >= next_look:
                next_look = time.monotonic() + GUARD_EVERY_SECONDS
                reason = memory()
                if reason:
                    raise MemoryStop(
                        f"Stopped to protect this Mac: {reason}. Close other programs and free some disk, then try again."
                    )
            time.sleep(POLL_SECONDS)
    finally:
        kill_group(process)
        reader.join(timeout=5)
        for pipe in (process.stdin, process.stdout):
            if pipe:
                pipe.close()
    return Run(process.returncode, list(tail))


def run_with_one_retry(
    run: Callable[[], Run],
    say: Callable[[str], None] = lambda _message: None,
    before_retry: Callable[[], None] = lambda: None,
) -> None:
    """A Metal timeout is retried once: measured 1 run in 5, during prompt encoding, and the next attempt ran.

    It fails early and costs little, and memory pressure is the likely cause (an inference: swap grew 8.6 GB in that
    run). Any other failure repeats itself, so it is reported at once.
    """
    first = run()
    if first.code == 0:
        return
    if not first.timed_out_on_gpu:
        raise GenerationFailed(describe_failure(first))
    say("Metal reported a GPU timeout. Trying once more.")
    before_retry()
    second = run()
    if second.code == 0:
        say("The second attempt worked.")
        return
    if second.timed_out_on_gpu:
        raise GenerationFailed(
            "The clip failed twice with a Metal GPU timeout. This Mac is probably short of memory: "
            "close other programs and try again."
        )
    raise GenerationFailed(describe_failure(second))


def describe_failure(run: Run) -> str:
    """What the child said last, without its progress bars."""
    said = [line for line in run.tail if "%|" not in line and not line.startswith("[estimate]")]
    last = said[-1] if said else "it said nothing"
    if run.code < 0 or run.code == 128 + 9:
        return f"The generation was killed from outside (exit {run.code}), which on a Mac usually means it ran out of memory. Last output: {last}"
    return f"The generation failed (exit {run.code}): {last}"


def child_environment(base: dict[str, str], bin_dirs: Sequence[Path]) -> dict[str, str]:
    """ComfyUI's environment minus what would make the child load ComfyUI's packages."""
    env = {key: value for key, value in base.items() if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP")}
    env["PATH"] = os.pathsep.join([*(str(path) for path in bin_dirs), base.get("PATH", "")])
    # The weights are on disk and nothing may be fetched behind the person's back: a missing file has to fail.
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    return env


STAGE_STEP = re.compile(r"(\d+)/(\d+) \[")


class Progress:
    """Turns the child's lines into a percentage that never goes back. Two denoising passes share the middle."""

    def __init__(self, report: Callable[[int], None]) -> None:
        self.report = report
        self.percent = 0
        self.passes = 0
        self.last_step = 0

    def __call__(self, line: str) -> None:
        if line.startswith("[Encoding prompt] done"):
            self._to(15)
        elif "Encoding prompt" in line:
            self._to(5)
        elif "Denoising" in line:
            match = STAGE_STEP.search(line)
            if not match:
                return
            step, total = int(match.group(1)), int(match.group(2))
            if step < self.last_step:
                self.passes += 1
            self.last_step = step
            # Stage 1 takes 15 to 70, stage 2 70 to 90.
            low, high = (15, 70) if self.passes == 0 else (70, 90)
            self._to(low + (high - low) * step // max(total, 1))
        elif "Decoding video" in line:
            self._to(92)

    def _to(self, percent: int) -> None:
        if percent > self.percent:
            self.percent = percent
            self.report(percent)


def python_of(venv: Path) -> Path:
    return venv / "bin" / "python"
