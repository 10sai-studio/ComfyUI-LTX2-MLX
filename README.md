# ComfyUI-LTX2-MLX

ComfyUI nodes that make LTX-2.5 clips with [`ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx) on Apple Silicon, with
`--distilled --low-ram`, in a process of their own:

- **Around a given sound** (`a2v`): the sound stays as it is, and the mouth follows it.
- **From text** and **from a picture** (`generate`, the second with `--image <file> 0 1.0`): the clip comes with its own sound.

Written for [10sai](https://github.com/10sai-studio/10sai), which installs it, and it works on any ComfyUI that does the same.
Width and height are snapped to multiples of 64, and the length to LTX's 8n+1 frames.

**Experimental.** On a Mac mini M6 with 16 GB and Chrome open and idle:

| Mode | Three clips in a row | Took | Memory available, lowest |
|---|---|---|---|
| Around a sound | 6 s at 448x768 | 202, 204, 235 s | 15% |
| From text | 4, 6, 6 s at 832x512 | 152, 224, 223 s | 19% |
| From a picture | 4, 6, 6 s at 832x512 | 163, 244, 243 s | 23% |

Each run started from 62 to 76% available. The lowest point of the text runs was one 2 s sample at the first clip's load,
under critical pressure; it went back to 49% in the next one. Swap grew by 1.3 GB over the three text clips and by 0.2 GB
over the three picture clips. Nothing larger than 832x512, longer than 6 s or started below 62% has been run.

**Gemma's tower is read one layer at a time.** `ltx-2-mlx` evaluates it in one lazy graph and `--low-ram` does not reach it,
so the node runs `stream_text_encoder.py`, which replaces that one method and then calls the program. Alone at length 1024,
the encoder went from a peak of 15.1 GB and 1% available to 7.5 GB and 18%. A clip with the same seed came out with the same
frames and the same audio. One process can encode once, which is all distilled `a2v` and `generate` do; a second call fails saying so.

**The prompt has to ask for the mouth.** Over music, "relaxed face, head still" left it shut, and "lips and jaw moving
clearly with the words" moved it, on one song, one seed and one face.

**A memory guard stops it before it takes the machine down.** It refuses to start with under 10 GB of free disk, over
12 GB of swap, critical memory pressure or under 25% of memory available, and kills the run under 5 GB of free disk, over 20 GB of swap or after 30 s of
critical pressure. Every run logs what it reached. The limits are not calibrated against a real crash yet.

## What it expects

- `ltx-2-mlx` in `.venv` inside this folder, on its own environment and never ComfyUI's, with `imageio-ffmpeg`.
- The 8-bit pack (`dgrauet/ltx-2.5-mlx-q8`) at `<models>/ltx-mlx/ltx-2.5-mlx-q8`, 42 GiB.

`prove_environment.py` checks that the machine can run it, and that the pinned program still gives the same numbers through
the replaced method, on a tiny tower. `python -m unittest discover tests` needs numpy only.

MIT licence. `ltx-2-mlx` is MIT, the weights are under the LTX-2.x Community License, and the ffmpeg `imageio-ffmpeg`
downloads is GPL.
