import numpy as np

from .._shared.utils import check_nD
from ..filters import sobel


def _image_energy(image):
    """Compute the edge-based energy of a 2D grayscale or 3D RGB image."""
    if image.ndim == 3:
        channel_energies = np.stack(
            [sobel(image[..., channel]) for channel in range(image.shape[2])],
            axis=-1,
        )
        return np.linalg.norm(channel_energies, axis=-1)
    return sobel(image)


def _minimum_energy_seam(energy, protected):
    """Find the lowest-cost vertical seam.

    A vertical seam connects the top and bottom image borders and contains
    exactly one pixel per row. Pixels flagged by ``protected`` are given an
    infinite cost so that the returned seam never crosses them.
    """
    rows, cols = energy.shape
    cost = np.where(protected, np.inf, energy.astype(np.float64, copy=True))

    for row in range(1, rows):
        previous = cost[row - 1]
        left = np.empty(cols)
        right = np.empty(cols)
        left[0] = np.inf
        left[1:] = previous[:-1]
        right[-1] = np.inf
        right[:-1] = previous[1:]
        cost[row] += np.minimum(previous, np.minimum(left, right))

    seam = np.empty(rows, dtype=np.intp)
    seam[-1] = np.argmin(cost[-1])
    if not np.isfinite(cost[-1, seam[-1]]):
        return None
    for row in range(rows - 1, 0, -1):
        column = seam[row]
        start = max(column - 1, 0)
        stop = min(column + 1, cols - 1)
        seam[row - 1] = start + np.argmin(cost[row - 1, start : stop + 1])
    return seam


def _remove_vertical_seam(image, seam):
    """Remove one vertical seam from a 2D or 3D image."""
    rows, cols = image.shape[:2]
    keep_mask = np.ones((rows, cols), dtype=bool)
    keep_mask[np.arange(rows), seam] = False
    if image.ndim == 3:
        return image[keep_mask].reshape(rows, cols - 1, image.shape[2])
    return image[keep_mask].reshape(rows, cols - 1)


def _carve_vertical(image, mask, num_seams):
    """Remove ``num_seams`` vertical seams, keeping protected pixels intact."""
    for _ in range(num_seams):
        energy = _image_energy(image)
        seam = _minimum_energy_seam(energy, mask)
        if seam is None:
            raise ValueError(
                "No valid seam can be found without removing a pixel of the "
                "protected region; the target size conflicts with the "
                "protection mask and the protected content cannot be "
                "preserved. Choose a larger target size or a smaller "
                "protection mask."
            )
        image = _remove_vertical_seam(image, seam)
        mask = _remove_vertical_seam(mask, seam)
    return image, mask


def seam_carve(image, output_shape, mask=None):
    """Reduce the size of an image with content-aware seam carving.

    Seams are connected paths of low-energy (unimportant) pixels that run
    from one image border to the opposite one. Repeatedly removing the
    lowest-energy seam shrinks the image while leaving visually important
    structures untouched, instead of uniformly scaling or center-cropping
    the image [1]_.

    Pixels marked in ``mask`` are protected: no seam is ever allowed to
    pass through them, so protected objects keep their original pixels and
    shape. The image is only reduced in size; upscaling is not supported.

    Parameters
    ----------
    image : ndarray
        Input image. Supported inputs are 2D grayscale images
        ``(rows, cols)`` or 3D RGB images ``(rows, cols, 3)``. The input is
        never modified.
    output_shape : tuple of int
        Target shape as ``(rows, cols)``. Each value must be a positive
        integer no larger than the corresponding input dimension.
    mask : ndarray of bool, optional
        Protection mask with shape ``(rows, cols)``. Pixels that are ``True``
        belong to protected content and cannot be removed. If satisfying
        ``output_shape`` would require removing a protected pixel, a
        ``ValueError`` is raised rather than degrading the protected region.

    Returns
    -------
    carved : ndarray
        Image with shape ``output_shape`` (plus the color channels of the
        input) and the same dtype as ``image``.

    Raises
    ------
    ValueError
        If ``image`` is neither a 2D grayscale nor an ``(rows, cols, 3)``
        RGB image, if ``output_shape`` requests upscaling or has the wrong
        length, if ``mask`` does not match the image shape, or if the
        protection mask makes the requested target size unreachable.

    References
    ----------
    .. [1] S. Avidan and A. Shamir, "Seam carving for content-aware image
           resizing", ACM Transactions on Graphics, vol. 26, no. 3, 2007.
           :DOI:`10.1145/1276377.1276390`

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import data
    >>> from skimage.transform import seam_carve
    >>> image = data.astronaut()
    >>> mask = np.zeros(image.shape[:2], dtype=bool)
    >>> mask[50:300, 150:350] = True  # protect the astronaut's face and body
    >>> carved = seam_carve(image, (image.shape[0], image.shape[1] - 80),
    ...                     mask=mask)
    >>> carved.shape
    (512, 432, 3)
    """
    check_nD(image, (2, 3), arg_name='image')
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(
            "Only 2D grayscale images and 3D RGB images with 3 channels are "
            f"supported; got an image of shape {image.shape}."
        )

    output_shape = tuple(output_shape)
    if len(output_shape) != 2:
        raise ValueError(
            "`output_shape` must be a (rows, cols) tuple; got "
            f"{output_shape}."
        )
    rows, cols = image.shape[:2]
    target_rows, target_cols = output_shape
    if not all(
        isinstance(dimension, (int, np.integer)) for dimension in output_shape
    ):
        raise ValueError("`output_shape` must contain integer values.")
    if target_rows <= 0 or target_cols <= 0:
        raise ValueError("`output_shape` values must be positive integers.")
    if target_rows > rows or target_cols > cols:
        raise ValueError(
            "seam_carve only reduces image size; target shape "
            f"{output_shape} exceeds input shape {(rows, cols)}."
        )

    if mask is None:
        protected = np.zeros((rows, cols), dtype=bool)
    else:
        protected = np.asarray(mask, dtype=bool)
        if protected.shape != (rows, cols):
            raise ValueError(
                f"`mask` shape {protected.shape} does not match image shape "
                f"{(rows, cols)}."
            )
    # Work on copies so the caller's arrays are never modified in place.
    working_image = image.copy()
    working_mask = protected.copy()

    # A vertical seam removes one column per row, hence at most
    # (cols - protected columns in the most-protected row) columns can be
    # removed. The same bound applied to the transposed image governs rows.
    max_protected_per_row = int(working_mask.sum(axis=1).max())
    if (cols - target_cols) > cols - max_protected_per_row:
        raise ValueError(
            f"Cannot reduce width from {cols} to {target_cols}: at least "
            f"{max_protected_per_row} column(s) per row are protected. "
            "Choose a larger target width or reduce the protection mask."
        )
    max_protected_per_col = int(working_mask.sum(axis=0).max())
    if (rows - target_rows) > rows - max_protected_per_col:
        raise ValueError(
            f"Cannot reduce height from {rows} to {target_rows}: at least "
            f"{max_protected_per_col} row(s) per column are protected. "
            "Choose a larger target height or reduce the protection mask."
        )

    # Carve width first (vertical seams), then height (horizontal seams).
    working_image, working_mask = _carve_vertical(
        working_image, working_mask, cols - target_cols
    )
    if working_image.ndim == 3:
        transposed_image = np.swapaxes(working_image, 0, 1)
    else:
        transposed_image = working_image.T
    transposed_mask = working_mask.T
    transposed_image, transposed_mask = _carve_vertical(
        transposed_image, transposed_mask, rows - target_rows
    )
    if transposed_image.ndim == 3:
        working_image = np.swapaxes(transposed_image, 0, 1)
    else:
        working_image = transposed_image.T

    return working_image.astype(image.dtype, copy=False)
