# ComfyUI-LTX2-MLX

A ComfyUI node that makes a clip around a given sound with [`ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx)
(`a2v --distilled --low-ram`) on Apple Silicon: the sound stays as it is, and the mouth follows it. Written for
[10sai](https://github.com/10sai-studio/10sai), which installs it, and it works on any ComfyUI that does the same.

**Experimental.** On a Mac mini M6 with 16 GB, 6 s at 448x768 takes about 225 s and peaks at 15.8 GB.

**The prompt has to ask for the mouth.** Over music, "relaxed face, head still" left it shut, and "lips and jaw moving
clearly with the words" moved it, on one song, one seed and one face.

**A memory guard stops it before it takes the machine down.** It refuses to start with under 10 GB of free disk, over
12 GB of swap, critical memory pressure or under 25% of memory available, and kills the run under 5 GB of free disk, over 20 GB of swap or after 30 s of
critical pressure. Every run logs what it reached. The limits are not calibrated against a real crash yet.

## What it expects

- `ltx-2-mlx` in `.venv` inside this folder, on its own environment and never ComfyUI's, with `imageio-ffmpeg`.
- The int4 pack (`dgrauet/ltx-2.5-mlx-q4`) at `<models>/ltx-mlx/ltx-2.5-mlx-q4`.

`prove_environment.py` checks that the machine can run it. `python -m unittest discover tests` needs numpy only.

MIT licence. `ltx-2-mlx` is MIT, the weights are under the LTX-2.x Community License, and the ffmpeg `imageio-ffmpeg`
downloads is GPL.
