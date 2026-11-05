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
