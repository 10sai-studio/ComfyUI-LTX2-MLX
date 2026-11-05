"""Runs `ltx-2-mlx` with Gemma's tower read one layer at a time, and gets the same numbers.

The tower is 9.8 GB and the program evaluates it in one lazy graph, so all of it sits in memory at once beside the 3.8 GB
connector. On 16 GB with a browser open that reached 1% available in 7 s. `--low-ram` streams the transformer and nothing
streams this, so the method is replaced here rather than forking a program pinned by its lockfile.
"""
import sys

import mlx.core as mx
from ltx_core_mlx.text_encoders.gemma import gemma4
from ltx_pipelines_mlx.cli import main as program

ALREADY_READ = (
    "Gemma's tower was read layer by layer and its weights are gone, so one process can encode one prompt. "
    "`stream_text_encoder.py` needs changing if this mode of ltx-2-mlx now encodes twice."
)


def streamed_call(self, input_ids, attention_mask=None):
    if self.embed_tokens is None:
        raise RuntimeError(ALREADY_READ)
    batch, seq_len = input_ids.shape
    hidden_states = self.embed_tokens(input_ids)
    hidden_states = hidden_states * mx.array(self.embed_scale, dtype=hidden_states.dtype)
    mx.eval(hidden_states)
    self.embed_tokens = None
    mx.clear_cache()
    position_ids = mx.broadcast_to(mx.arange(seq_len)[None, :], (batch, seq_len))
    rope_tables = {kind: self._rotary_by_type[kind](position_ids) for kind in sorted(set(self.config.layer_types))}
    masks = {
        gemma4._FULL_ATTENTION: gemma4.build_attention_mask(seq_len, sliding_window=None, padding_mask=attention_mask, dtype=hidden_states.dtype),
        gemma4._SLIDING_ATTENTION: gemma4.build_attention_mask(seq_len, sliding_window=self.config.sliding_window, padding_mask=attention_mask, dtype=hidden_states.dtype),
    }
    states = [hidden_states]
    for index in range(len(self.layers)):
        kind = self.config.layer_types[index]
        cos, sin = rope_tables[kind]
        hidden_states = self.layers[index](hidden_states, cos, sin, masks[kind])
        mx.eval(hidden_states)
        self.layers[index] = None
        mx.clear_cache()
        states.append(hidden_states)
    return states


def install() -> None:
    gemma4.Gemma4TextModel.__call__ = streamed_call


if __name__ == "__main__":
    install()
    sys.argv = ["ltx-2-mlx", *sys.argv[1:]]
    sys.exit(program())
