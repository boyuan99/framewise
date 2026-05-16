"""Generate a synthetic TIFF stack and an MP4 video for testing loaders.

Reuses the make_stack() helper from make_test_hdf5.py.
"""

from pathlib import Path

import imageio
import numpy as np
import tifffile

from make_test_hdf5 import make_stack


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "test_data"
    out_dir.mkdir(exist_ok=True)

    # --- TIFF stack ---
    tiff_path = out_dir / "stack.tif"
    stack = make_stack(t=150, h=256, w=256, seed=42)
    tifffile.imwrite(tiff_path, stack)
    print(f"Wrote {tiff_path} — shape {stack.shape}")

    # --- MP4 video (all-keyframe H.264 for smooth seek) ---
    mp4_path = out_dir / "video.mp4"
    stack8 = (make_stack(t=120, h=240, w=320, seed=7) // 256).astype(np.uint8)
    # Convert to RGB by stacking the grayscale channel
    rgb = np.repeat(stack8[..., None], 3, axis=-1)
    writer = imageio.get_writer(
        mp4_path,
        fps=30,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-g", "1", "-pix_fmt", "yuv420p"],
    )
    for frame in rgb:
        writer.append_data(frame)
    writer.close()
    print(f"Wrote {mp4_path} — {len(rgb)} frames @ 30fps, all-keyframe H.264")


if __name__ == "__main__":
    main()
