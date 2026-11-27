"""
========================
Stitch a simple panorama
========================

This example stitches three partially overlapping crops into one panorama.
Only adjacent images share point correspondences, so the third image is
connected to the first image indirectly. The crops also have different
brightness levels.
"""

import matplotlib.pyplot as plt
import numpy as np

from skimage import data, metrics
from skimage.transform import stitch_images
from skimage.util import img_as_float

scene = img_as_float(data.moon())

offsets = [20, 120, 220]
crop_width = 280
crop_rows = slice(80, 320)
images = [
    scene[crop_rows, offset : offset + crop_width].astype(float) * gain
    for offset, gain in zip(offsets, (1.0, 0.85, 1.15))
]

# Correspondences are given as (x, y) coordinates. The middle image bridges
# the first and third images; no direct matches are needed between images 0
# and 2.
points_01 = np.array(
    [[150, 60], [200, 100], [250, 140], [280, 180], [180, 200], [220, 40]],
    dtype=float,
)
points_12 = np.array(
    [[260, 60], [310, 100], [360, 140], [390, 180], [290, 200], [330, 40]],
    dtype=float,
)
correspondences = [
    (0, 1, points_01 - [offsets[0], 0], points_01 - [offsets[1], 0]),
    (1, 2, points_12 - [offsets[1], 0], points_12 - [offsets[2], 0]),
]

panorama, coverage = stitch_images(
    images, correspondences, point_order='xy', rng=0
)

# Naively averaging the images at the same origin preserves none of the
# relative camera positions and produces obvious ghosting.
naive_overlay = np.mean(images, axis=0)

expected = scene[crop_rows, offsets[0] : offsets[-1] + crop_width]
naive_mse = metrics.mean_squared_error(
    expected[:, : naive_overlay.shape[1]], naive_overlay
)
stitched_mse = metrics.mean_squared_error(
    expected, panorama[: expected.shape[0], : expected.shape[1]]
)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].imshow(naive_overlay, cmap='gray', vmin=0, vmax=1)
axes[0].set_title(f'Direct overlay (MSE={naive_mse:.3f})')
axes[0].set_axis_off()

axes[1].imshow(panorama, cmap='gray', vmin=0, vmax=1)
axes[1].set_title(f'Stitched panorama (MSE={stitched_mse:.3f})')
axes[1].set_axis_off()

fig.tight_layout()
plt.show()

# sphinx_gallery_thumbnail_number = 2
