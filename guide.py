"""What the IC-LoRA path of `ltx-2-mlx` needs changed to edit a clip on a 16 GB Mac: the guide encoded in tiles, and
read without `ffprobe`.

The guide is the edited first frame followed by the clip. `ltx-2-mlx` encodes it in one call: 25 frames at 832x512
peaked at 7.84 GiB, so a 4 s guide (97 frames) needs about 30 GiB and the Mac stops. The encoder already has
`tiled_encode`, which retake and HDR use, but the IC-LoRA path calls `encode` directly, so that path is handed an
encoder whose `encode` is the tiled one. A tile's peak follows its volume, not the clip's, so the clip can be any length.

`ltx-2-mlx` probes the guide with `ffprobe`, which a Mac does not come with and `imageio-ffmpeg` does not carry. The
probe is answered by `imageio-ffmpeg` instead, which the environment already has for `ffmpeg`.
"""
import mlx.core as mx
from ltx_core_mlx.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig, prepare_tiles_for_encoding
from ltx_core_mlx.utils import ffmpeg
from ltx_pipelines_mlx import ic_lora, iclora_utils
from ltx_pipelines_mlx.utils import media_io

# Measured on the 4 s guide at 832x512 (M6, q8 VAE): 7.12 GiB peak instead of about 30. Against the untiled latent, the
# time seams alone move it 2.4% and the space seams alone 5.3%; H.264 at CRF 18 moves it 7.1%, so the tiles change the
# guide less than the clip's own compression did. A 16-frame overlap left 10% at every time seam: each tile starts
# without the frames before it, and the VAE is causal.
GUIDE_TILES = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=384, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=48),
)


class TiledEncoder:
    """The loaded encoder, with `encode` swapped for the tiled one. Everything else is the encoder's own."""

    def __init__(self, encoder, tiles: TilingConfig = GUIDE_TILES):
        self._encoder = encoder
        self._tiles = tiles

    def encode(self, video: mx.array) -> mx.array:
        # One tile is the untiled computation, so a guide that fits keeps its exact latent.
        if len(prepare_tiles_for_encoding(video.shape, self._tiles)) == 1:
            return self._encoder.encode(video)
        # The blend buffers are fp32; back to the encoder's bf16, as retake does.
        return self._encoder.tiled_encode(video, self._tiles).astype(mx.bfloat16)

    def __getattr__(self, name):
        return getattr(self._encoder, name)


def probe_without_ffprobe(video_path: str) -> ffmpeg.VideoInfo:
    import imageio_ffmpeg

    frames = imageio_ffmpeg.read_frames(video_path)
    meta = next(frames)
    frames.close()
    count, seconds = imageio_ffmpeg.count_frames_and_secs(video_path)
    width, height = meta["size"]
    return ffmpeg.VideoInfo(
        width=width, height=height, num_frames=count, fps=float(meta["fps"]), duration=seconds,
        has_audio=bool(meta.get("audio_codec")),
    )


def install(tiles: TilingConfig = GUIDE_TILES) -> None:
    untiled = iclora_utils.append_ic_lora_reference_video_conditionings

    def tiled(conditionings, video_conditioning, *, video_encoder, **rest):
        return untiled(conditionings, video_conditioning, video_encoder=TiledEncoder(video_encoder, tiles), **rest)

    # Each module imported these by name, so every name is replaced.
    iclora_utils.append_ic_lora_reference_video_conditionings = tiled
    ic_lora.append_ic_lora_reference_video_conditionings = tiled
    for module in (ffmpeg, media_io, iclora_utils):
        module.probe_video_info = probe_without_ffprobe
