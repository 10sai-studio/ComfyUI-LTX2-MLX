# ComfyUI-LTX2-MLX

ComfyUI nodes that make LTX-2.5 clips with [`ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx) on Apple Silicon, with
`--low-ram`, in a process of their own:

- **Around a given sound** (`a2v`): the sound stays as it is, and the mouth follows it.
- **From text** and **from a picture** (`generate`, the second with `--image <file> 0 1.0`): the clip comes with its own sound.
- **A clip edited from its first frame** (`ic-lora --single-stage` with the Ripple LoRA): the edit made on the first frame is
  carried through the clip, which keeps its motion, camera and timing. It comes back without sound, as the CUDA level does.

Written for [10sai](https://github.com/10sai-studio/10sai), which installs it, and it works on any ComfyUI that does the same.
Width and height are snapped to multiples of 64, and the length to LTX's 8n+1 frames.

**Experimental.** On a Mac mini M6 with 16 GB and Chrome open and idle, with the 8-bit pack:

| Mode | Three clips in a row | Took | Memory available, lowest |
|---|---|---|---|
| Around a sound | 6 s at 448x768 | 214, 214, 214 s | 31% |
| From text | 4, 6, 6 s at 832x512 | 173, 244, 244 s | 26% |
| From a picture | 4, 6, 6 s at 832x512 | 183, 254, 254 s | 26% |
| A Ripple edit | 4 s at 832x512 | 631, 638, 622 s | 13-15% |

Each run started from 70 to 76% available, and swap grew by 0.3 GB at most, 2.0 GB for an edit. With the int4 pack the same runs were about
10% faster and went down to 15-23%. Nothing larger than 832x512, longer than 6 s or started below 62% has been run.

**Gemma's tower is read one layer at a time.** `ltx-2-mlx` evaluates it in one lazy graph and `--low-ram` does not reach it,
so the node runs `stream_text_encoder.py`, which replaces that one method and then calls the program. Alone at length 1024,
the encoder went from a peak of 15.1 GB and 1% available to 7.5 GB and 18%. A clip with the same seed came out with the same
frames and the same audio. One process can encode once, which is all distilled `a2v` and `generate` do; a second call fails saying so.

**An edit encodes its guide in tiles.** The guide is the edited first frame and then the clip, and `ltx-2-mlx` encodes it
in one call: 25 frames at 832x512 peaked at 7.84 GiB, so a 4 s edit needed about 30 GiB and stopped the Mac at 0% available.
`guide.py` hands that call the encoder's own `tiled_encode`, in tiles of 64 frames overlapping 48 and 384 px overlapping 64:
7.12 GiB on the 4 s guide. Against the untiled latent, the time seams alone move it 2.4% and the space seams alone 5.3%,
where H.264 at CRF 18 moves it 7.1%. A 16-frame overlap left 10% at every time seam, because the VAE is causal and a tile
starts without the frames before it. `guide.py` also answers `ltx-2-mlx`'s `ffprobe` with `imageio-ffmpeg`, since a Mac
has no `ffprobe`. The node builds the guide as the CUDA level does: the clip's aspect at the asked shorter side, the clip's
own rate, its frames 0 to N-2 after the edited one, cut to the aspect from the centre.

**The prompt has to ask for the mouth.** Over music, "relaxed face, head still" left it shut, and "lips and jaw moving
clearly with the words" moved it, on one song, one seed and one face.

**A memory guard stops it before it takes the machine down.** It refuses to start with under 10 GB of free disk, over
12 GB of swap, critical memory pressure or under 25% of memory available, and kills the run under 5 GB of free disk, over 20 GB of swap or after 30 s of
critical pressure. Every run logs what it reached. The limits are not calibrated against a real crash yet.

## What it expects

- `ltx-2-mlx` in `.venv` inside this folder, on its own environment and never ComfyUI's, with `imageio-ffmpeg`.
- The 8-bit pack (`dgrauet/ltx-2.5-mlx-q8`) at `<models>/ltx-mlx/ltx-2.5-mlx-q8`, 42 GiB.
- For an edit, the Ripple LoRA (`WepeNerd/LTX-Ripple`, `LTX25_Ripple_v11.safetensors`) in ComfyUI's `loras`, picked on the node.

`prove_environment.py` checks that the machine can run it, and that the pinned program still gives the same numbers through
the replaced method, on a tiny tower. `python -m unittest discover tests` needs numpy only.

MIT licence. `ltx-2-mlx` is MIT, the weights are under the LTX-2.x Community License, and the ffmpeg `imageio-ffmpeg`
downloads is GPL.
