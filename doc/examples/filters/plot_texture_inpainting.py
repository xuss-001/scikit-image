"""
=========================================
Texture-aware inpainting with source masks
=========================================

Smooth inpainting such as :func:`skimage.restoration.inpaint_biharmonic`
reconstructs a smooth surface and can erase periodic texture or lines that
cross a damaged region.  :func:`skimage.restoration.inpaint_texture` instead
fills the region with small, overlapping patches sampled from intact parts of
the same image.

The example below uses a regular striped texture crossed by a horizontal line.
The line and the stripes are interrupted by the mask.  The biharmonic result
is smooth inside the gap, while the exemplar-based result continues both the
stripe pattern and the horizontal line.
"""

import matplotlib.pyplot as plt
import numpy as np

from skimage.restoration import inpaint_biharmonic, inpaint_texture


height, width = 48, 64
yy, xx = np.mgrid[:height, :width]

# A periodic striped texture with a horizontal line running through it.
image = ((xx // 4) % 2).astype(float) * 0.55 + 0.2
image[22:24, :] = 0.05

# Damage a rectangular region that contains part of the line.
mask = np.zeros((height, width), dtype=bool)
mask[17:31, 28:37] = True
damaged = image.copy()
damaged[mask] = 0

# Limit source material to the left part of the image.  The masked pixels are
# excluded automatically even where source_mask is True.
source_mask = np.ones((height, width), dtype=bool)
source_mask[:, 52:] = False

smooth = inpaint_biharmonic(damaged, mask)
textured = inpaint_texture(
    damaged, mask, source_mask=source_mask, patch_size=9
)

fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
panels = [
    (image, 'Original texture'),
    (damaged, 'Masked input'),
    (source_mask.astype(float), 'Allowed source material'),
    (smooth, 'Biharmonic (smooth)'),
    (textured, 'Exemplar texture inpainting'),
]

for ax, (panel, title) in zip(axes.ravel()[:5], panels):
    ax.imshow(panel, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_axis_off()

profile_ax = axes.ravel()[-1]
profile_ax.plot(image[22], label='Original')
profile_ax.plot(smooth[22], label='Biharmonic')
profile_ax.plot(textured[22], label='Texture')
profile_ax.set_title('Line profile (row 22)')
profile_ax.set_xlabel('column')
profile_ax.legend(fontsize='small')

fig.tight_layout()
plt.show()
