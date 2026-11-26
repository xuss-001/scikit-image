"""
===============================
Content-aware image resizing
===============================

When an image has to fit a banner of a different aspect ratio, plain
resizing stretches the content and center cropping may cut off the main
subject. Seam carving instead removes connected paths of low-energy pixels
("seams"), so that visually important structures are kept while the
unimportant background is compressed.

This example compares the three approaches and shows how a protection mask
guarantees that the marked subject is neither removed nor distorted.
"""

import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.transform import resize, seam_carve

image = data.astronaut()
height, width = image.shape[:2]

# The banner is narrower than the original image.
target_shape = (height, width - 120)

# Naive resizing changes the aspect ratio and squashes the astronaut.
squashed = resize(
    image, target_shape, anti_aliasing=True, preserve_range=True
).astype(image.dtype)

# Center cropping keeps the aspect ratio but removes content on both sides.
cut = (width - target_shape[1]) // 2
center_cropped = image[:, cut : cut + target_shape[1]]

# The mask marks the astronaut (face and body) as protected content: no seam
# is allowed to cross these pixels.
mask = np.zeros((height, width), dtype=bool)
mask[30:460, 130:370] = True
carved = seam_carve(image, target_shape, mask=mask)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title(f"Original ({width} x {height})")

ax[1].imshow(squashed)
ax[1].set_title(f"Resized, distorted ({carved.shape[1]} x {height})")

ax[2].imshow(center_cropped)
ax[2].set_title(f"Center cropped, content lost ({carved.shape[1]} x {height})")

ax[3].imshow(carved)
ax[3].set_title(
    f"Seam carved with protection mask ({carved.shape[1]} x {height})"
)

for axis in ax:
    axis.axis("off")
plt.tight_layout()

# Same target width reached without upscaling and without touching the mask.
print("target size:", target_shape)
print("seam-carved size:", carved.shape)
print("protected pixels in input mask:", int(mask.sum()))

plt.show()
