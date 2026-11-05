"""Run by the installer with the pack's own Python: can this machine do the three things a clip needs?

Metal has to answer through MLX, ffmpeg has to encode with libx264, which `ltx-2-mlx` hard-codes, and the program has to
still be what `stream_text_encoder.py` replaces a method of. Reading a version off any of them would say yes on a machine
that then fails after a 30 GB download.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import mlx.core as mx
from ltx_core_mlx.text_encoders.gemma import gemma4
from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig

import stream_text_encoder


def tiny_tower() -> gemma4.Gemma4TextModel:
    config = Gemma4TextConfig(
        gemma_version=None, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, head_dim=8, global_head_dim=8,
        num_key_value_heads=2, num_global_key_value_heads=2, intermediate_size=32, vocab_size=32, rms_norm_eps=1e-6,
        sliding_window=2, layer_types=(gemma4._SLIDING_ATTENTION, gemma4._FULL_ATTENTION), attention_k_eq_v=False,
        rope_local_theta=10000.0, rope_global_theta=10000.0, partial_rotary_factor=1.0, pad_token_id=0,
    )
    return gemma4.Gemma4TextModel(config)


def prove_the_tower_reads_the_same() -> str | None:
    """The replaced method has to give the program's own numbers, layer for layer, and refuse a second call."""
    with mx.stream(mx.cpu):
        tower = tiny_tower()
        ids = mx.array([[0, 1, 2, 3]])
        mask = mx.array([[0, 1, 1, 1]])
        theirs = tower(ids, attention_mask=mask)
        mx.eval(theirs)
        ours = stream_text_encoder.streamed_call(tower, ids, attention_mask=mask)
        if len(ours) != len(theirs) or not all(mx.array_equal(a, b).item() for a, b in zip(ours, theirs)):
            return "the layer-by-layer tower no longer gives the same numbers as ltx-2-mlx"
        try:
            stream_text_encoder.streamed_call(tower, ids, attention_mask=mask)
        except RuntimeError:
            return None
        return "the layer-by-layer tower encoded twice without its weights"


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

    wrong = prove_the_tower_reads_the_same()
    if wrong:
        print(f"This ltx-2-mlx does not fit the node: {wrong}.", file=sys.stderr)
        return 1
    print("MLX on the GPU, ffmpeg encodes H.264, and the tower reads the same layer by layer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
