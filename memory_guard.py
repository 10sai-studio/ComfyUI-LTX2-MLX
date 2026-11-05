"""Stops a generation before it takes the machine down. Nothing here imports ComfyUI.

macOS has no memory limit one process can be held to: `RLIMIT_AS` is not enforced and mmap'd weights do not count as a
footprint. What a Mac has is what hangs when this goes wrong, so that is what gets watched: the kernel's own pressure
level, the swap, and the disk the swap lands on. A full disk under a growing swap is what freezes the machine.

The owner's Mac had to be restarted on 2026-11-05 with about 13 GB of disk free. These limits are NOT calibrated against
the moment of a crash, because nothing recorded it. What they rest on is healthy runs: swap reached 14.8 GB on one that
began with 6.8 GB already used, and 7.7 GB on one that began with none. Every run prints what it reached (`peak_line`).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

GIB = 1024**3
# Not to start: below this there is no room for the swap a normal run grows by (8 GB measured).
MIN_FREE_DISK_TO_START = 10 * GIB
MAX_SWAP_TO_START = 12 * GIB
# The kernel's figure for available memory, in percent, the one `memory_pressure` prints. The run takes all of it and
# gives the file-backed part back, so what it cannot start without is room for the part that is not: about 4 GB of 16.
MIN_AVAILABLE_PERCENT_TO_START = 25
# To stop: the disk is what freezes the machine, so this is the hard line.
MIN_FREE_DISK = 5 * GIB
MAX_SWAP = 20 * GIB
CRITICAL = 4
# A healthy 6 s clip sits at critical pressure twice, about 8 s each, loading the text encoder: lowest 12% available, on a
# run that finished in 224 s. So a stretch of critical is ordinary, and only one several times that long is not.
CRITICAL_SECONDS = 30


@dataclass
class Reading:
    free_disk: int
    swap: int
    pressure: int
    available_percent: int = 100


def stop_reason_at_start(reading: Reading) -> Optional[str]:
    if reading.free_disk < MIN_FREE_DISK_TO_START:
        return f"it needs {MIN_FREE_DISK_TO_START // GIB} GB of free disk for swap and there are {reading.free_disk / GIB:.1f} GB"
    if reading.swap > MAX_SWAP_TO_START:
        return f"the machine is already using {reading.swap / GIB:.1f} GB of swap"
    if reading.pressure >= CRITICAL:
        return "the machine is already under critical memory pressure"
    if reading.available_percent < MIN_AVAILABLE_PERCENT_TO_START:
        return f"only {reading.available_percent}% of its memory is available and it needs {MIN_AVAILABLE_PERCENT_TO_START}%"
    return None


class Watch:
    """`check` answers why to stop, or `None`. Critical pressure has to last: a moment of it is ordinary on 16 GB."""

    def __init__(self, read: Callable[[], Reading], now: Callable[[], float]) -> None:
        self.read = read
        self.now = now
        self.critical_since: Optional[float] = None
        self.peak_swap = 0
        self.lowest_disk: Optional[int] = None
        self.worst_pressure = 0

    def check(self) -> Optional[str]:
        reading = self.read()
        self.peak_swap = max(self.peak_swap, reading.swap)
        self.lowest_disk = reading.free_disk if self.lowest_disk is None else min(self.lowest_disk, reading.free_disk)
        self.worst_pressure = max(self.worst_pressure, reading.pressure)
        if reading.free_disk < MIN_FREE_DISK:
            return f"only {reading.free_disk / GIB:.1f} GB of disk left for swap"
        if reading.swap > MAX_SWAP:
            return f"swap reached {reading.swap / GIB:.1f} GB"
        if reading.pressure < CRITICAL:
            self.critical_since = None
            return None
        now = self.now()
        if self.critical_since is None:
            self.critical_since = now
        if now - self.critical_since >= CRITICAL_SECONDS:
            return f"critical memory pressure for {CRITICAL_SECONDS} s"
        return None

    def peak_line(self) -> str:
        disk = "unknown" if self.lowest_disk is None else f"{self.lowest_disk / GIB:.1f} GB"
        return f"Reached: swap {self.peak_swap / GIB:.1f} GB, lowest free disk {disk}, worst pressure level {self.worst_pressure}."


SWAP_USED = re.compile(r"used = ([0-9.]+)([MG])")


def parse_swap(text: str) -> int:
    match = SWAP_USED.search(text)
    if not match:
        raise ValueError(f"Could not read the swap from: {text!r}")
    return int(float(match.group(1)) * (GIB if match.group(2) == "G" else 1024**2))


def _sysctl(name: str) -> str:
    return subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True).stdout


def read_machine() -> Reading:
    return Reading(
        free_disk=shutil.disk_usage("/").free,
        swap=parse_swap(_sysctl("vm.swapusage")),
        pressure=int(_sysctl("kern.memorystatus_vm_pressure_level").strip()),
        available_percent=int(_sysctl("kern.memorystatus_level").strip()),
    )
