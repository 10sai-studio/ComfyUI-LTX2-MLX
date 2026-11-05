"""Run by the installer with the pack's own Python: can this machine do the two things a clip needs?

Metal has to answer through MLX, and ffmpeg has to encode with libx264, which `ltx-2-mlx` hard-codes. Reading a version
off either would say yes on a machine that then fails after a 30 GB download.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import mlx.core as mx


def main() -> int:
    total = mx.sum(mx.ones((8, 8)))
    mx.eval(total)
    if int(total.item()) != 64 or mx.default_device().type != mx.DeviceType.gpu:
        print(f"MLX did not run on the GPU: {mx.default_device()}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "one-frame.mp4"
        done = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.04",
             "-frames:v", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
            capture_output=True, text=True,
        )
        if done.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            print(f"ffmpeg could not encode H.264: {done.stderr.strip()}", file=sys.stderr)
            return 1
    print("MLX on the GPU, and ffmpeg encodes H.264.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
