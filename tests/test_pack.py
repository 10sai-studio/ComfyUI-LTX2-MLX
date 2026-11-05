"""Run with: python -m unittest discover tests. No ComfyUI, no GPU, numpy only."""
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import memory_guard
import stretch
import supervise


def fake(code: str, folder: str) -> list[str]:
    script = Path(folder) / "fake.py"
    script.write_text(code)
    return [sys.executable, str(script)]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class Stretch(unittest.TestCase):
    def test_frames_land_on_the_grid_and_never_short(self):
        self.assertEqual(stretch.frames_for(6.0, 24), 145)
        self.assertEqual(stretch.frames_for(4.0, 24), 97)
        self.assertEqual(stretch.frames_for(6.04, 24), 145)
        self.assertEqual(stretch.frames_for(6.05, 24), 153)

    def test_a_short_track_is_padded_with_silence(self):
        out = stretch.fit_stretch(np.ones((2, int(5.98 * 48000)), dtype=np.float32), 48000, 0, 145, 24)
        self.assertGreaterEqual(out.shape[1], int((145 / 24 + stretch.LATENT_SECONDS) * 48000))
        self.assertEqual(out[0, 0], 1.0)
        self.assertEqual(out[0, -1], 0.0)

    def test_a_stretch_starts_where_it_is_asked(self):
        wave = np.arange(48000 * 20, dtype=np.float32)[None, :]
        out = stretch.fit_stretch(wave, 48000, 10.0, 97, 24)
        self.assertEqual(out[0, 0], 480000)
        self.assertLess(out.shape[1], 48000 * 5)

    def test_a_start_past_the_end_says_so(self):
        with self.assertRaisesRegex(ValueError, "starts at 9.00"):
            stretch.fit_stretch(np.zeros((1, 48000)), 48000, 9.0, 97, 24)


class Retry(unittest.TestCase):
    def test_a_gpu_timeout_is_retried_once_and_said(self):
        runs = [supervise.Run(1, ["[METAL] Command buffer execution failed: GPU Timeout"]), supervise.Run(0, [])]
        said = []
        supervise.run_with_one_retry(lambda: runs.pop(0), say=said.append)
        self.assertEqual(runs, [])
        self.assertTrue(any("once more" in line for line in said))

    def test_a_second_timeout_says_so(self):
        timeout = supervise.Run(1, ["[METAL] Command buffer execution failed: GPU Timeout"])
        with self.assertRaisesRegex(supervise.GenerationFailed, "failed twice"):
            supervise.run_with_one_retry(lambda: timeout)

    def test_any_other_failure_is_not_retried(self):
        calls = []

        def run():
            calls.append(1)
            return supervise.Run(1, ["ValueError: nope", "x%| bar"])

        with self.assertRaisesRegex(supervise.GenerationFailed, "ValueError: nope"):
            supervise.run_with_one_retry(run)
        self.assertEqual(len(calls), 1)


class Children(unittest.TestCase):
    def run_child(self, code, should_stop=lambda: False):
        with tempfile.TemporaryDirectory() as folder:
            return supervise.run_once(fake(code, folder), sys.executable, dict(os.environ), Path(folder) / "log", should_stop)

    def test_its_output_and_code_come_back(self):
        run = self.run_child("import sys\nprint('hello', flush=True)\nsys.exit(3)")
        self.assertEqual(run.code, 3)
        self.assertEqual(run.tail, ["hello"])

    def test_cancelling_kills_the_child_and_its_own_children(self):
        with tempfile.TemporaryDirectory() as folder:
            pids = Path(folder) / "pids"
            code = (
                "import os, subprocess, time\n"
                "sub = subprocess.Popen(['sleep', '60'])\n"
                f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{sub.pid}}')\n"
                "time.sleep(60)\n"
            )
            with self.assertRaises(supervise.Cancelled):
                self.run_child(code, should_stop=lambda: pids.exists() and pids.stat().st_size > 0)
            for pid in pids.read_text().split():
                self.assertFalse(alive(int(pid)), f"{pid} survived")

    def test_killing_only_the_guard_takes_the_rest_with_it(self):
        with tempfile.TemporaryDirectory() as folder:
            pids = Path(folder) / "pids"
            code = (
                "import os, signal, subprocess, time\n"
                "sub = subprocess.Popen(['sleep', '60'])\n"
                f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{sub.pid}}')\n"
                "os.kill(os.getppid(), signal.SIGKILL)\n"
                "time.sleep(60)\n"
            )
            run = self.run_child(code)
            self.assertLess(run.code, 0)
            self.assertIn("killed from outside", supervise.describe_failure(run))
            for pid in pids.read_text().split():
                self.assertFalse(alive(int(pid)), f"{pid} survived")

    def test_a_dead_parent_leaves_nothing_running(self):
        with tempfile.TemporaryDirectory() as folder:
            pids = Path(folder) / "pids"
            child = fake(
                "import os, subprocess, time\n"
                "sub = subprocess.Popen(['sleep', '60'])\n"
                f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{sub.pid}}')\n"
                "time.sleep(60)\n",
                folder,
            )
            pack = Path(supervise.__file__).parent
            parent = subprocess.Popen([
                sys.executable, "-c",
                f"import sys, os; sys.path.insert(0, {str(pack)!r}); import supervise, pathlib;"
                f"supervise.run_once({child!r}, sys.executable, dict(os.environ), pathlib.Path({str(Path(folder) / 'log')!r}), lambda: False)",
            ])
            deadline = time.monotonic() + 15
            while not (pids.exists() and pids.stat().st_size > 0) and time.monotonic() < deadline:
                time.sleep(0.05)
            found = [int(pid) for pid in pids.read_text().split()]
            parent.send_signal(signal.SIGKILL)
            parent.wait()
            deadline = time.monotonic() + 5
            while any(alive(pid) for pid in found) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(any(alive(pid) for pid in found), "the child outlived its parent")


GIB = memory_guard.GIB


class Memory(unittest.TestCase):
    def watch(self, readings, times):
        readings, times = iter(readings), iter(times)
        return memory_guard.Watch(lambda: next(readings), lambda: next(times))

    def test_a_short_disk_or_a_heavy_swap_refuses_to_start(self):
        self.assertIn("free disk", memory_guard.stop_reason_at_start(memory_guard.Reading(9 * GIB, 0, 1)))
        self.assertIn("swap", memory_guard.stop_reason_at_start(memory_guard.Reading(50 * GIB, 13 * GIB, 1)))
        self.assertIn("critical", memory_guard.stop_reason_at_start(memory_guard.Reading(50 * GIB, 0, 4)))
        self.assertIn("available", memory_guard.stop_reason_at_start(memory_guard.Reading(50 * GIB, 0, 1, 20)))
        self.assertIsNone(memory_guard.stop_reason_at_start(memory_guard.Reading(13 * GIB, 7 * GIB, 1)))

    def test_what_a_normal_run_reached_does_not_stop_it(self):
        # Measured: 14.8 GB of swap during prompt encoding, on a run that finished, with 13 GB of disk free at the start.
        self.assertIsNone(self.watch([memory_guard.Reading(6 * GIB, int(14.8 * GIB), 2)], [0]).check())

    def test_the_disk_and_the_swap_stop_it_at_once(self):
        self.assertIn("disk", self.watch([memory_guard.Reading(4 * GIB, 0, 1)], [0]).check())
        self.assertIn("swap", self.watch([memory_guard.Reading(50 * GIB, 21 * GIB, 1)], [0]).check())

    def test_critical_pressure_has_to_last(self):
        # Measured on a healthy run: two stretches of about 8 s each.
        critical = memory_guard.Reading(50 * GIB, 0, memory_guard.CRITICAL)
        normal = memory_guard.Reading(50 * GIB, 0, 1)
        # A normal reading reads no clock, so three clock readings for four checks.
        watch = self.watch([critical, normal, critical, critical], [0, 20, 50])
        self.assertIsNone(watch.check())
        self.assertIsNone(watch.check())
        self.assertIsNone(watch.check())
        self.assertIn("critical", watch.check())

    def test_every_run_says_what_it_reached(self):
        watch = self.watch([memory_guard.Reading(6 * GIB, 15 * GIB, 2)], [0])
        watch.check()
        self.assertEqual(watch.peak_line(), "Reached: swap 15.0 GB, lowest free disk 6.0 GB, worst pressure level 2.")

    def test_the_swap_line_of_this_mac_is_read(self):
        self.assertEqual(memory_guard.parse_swap("total = 16384.00M  used = 6814.75M  free = 9569.25M  (encrypted)"), int(6814.75 * 1024**2))

    def test_this_machine_can_be_read(self):
        reading = memory_guard.read_machine()
        self.assertGreater(reading.free_disk, 0)
        self.assertIn(reading.pressure, (1, 2, 4))
        self.assertTrue(0 <= reading.available_percent <= 100)

    def test_a_stop_kills_the_group_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            pids = Path(folder) / "pids"
            code = (
                "import os, subprocess, time\n"
                "sub = subprocess.Popen(['sleep', '60'])\n"
                f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{sub.pid}}')\n"
                "time.sleep(60)\n"
            )
            calls = []

            def run():
                calls.append(1)
                return supervise.run_once(
                    fake(code, folder), sys.executable, dict(os.environ), Path(folder) / "log", lambda: False,
                    memory=lambda: "swap reached 21.0 GB" if pids.exists() and pids.stat().st_size > 0 else None,
                )

            with self.assertRaisesRegex(supervise.MemoryStop, "protect this Mac: swap reached"):
                supervise.run_with_one_retry(run)
            self.assertEqual(len(calls), 1)
            for pid in pids.read_text().split():
                self.assertFalse(alive(int(pid)), f"{pid} survived")


class Lines(unittest.TestCase):
    def test_progress_only_goes_forward_across_both_passes(self):
        seen = []
        progress = supervise.Progress(seen.append)
        for line in [
            "[Encoding prompt] ...",
            "[Encoding prompt] done in 30.6s",
            "Denoising (ancestral):  50%|##| 4/8 [00:22<00:22,  5.68s/it]",
            "Denoising (ancestral):  88%|##| 7/8 [00:39<00:05,  5.67s/it]",
            "Denoising (ancestral):  33%|##| 1/3 [00:10<00:20,  5.67s/it]",
            "[Decoding video + audio + muxing] ...",
        ]:
            progress(line)
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(seen[-1], 92)

    def test_the_child_cannot_see_comfyuis_packages(self):
        env = supervise.child_environment({"PATH": "/usr/bin", "PYTHONPATH": "/comfy", "VIRTUAL_ENV": "/comfy/venv"}, [Path("/pack/bin")])
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("VIRTUAL_ENV", env)
        self.assertTrue(env["PATH"].startswith("/pack/bin"))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
