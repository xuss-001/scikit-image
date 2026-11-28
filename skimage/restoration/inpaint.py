import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage import laplace

import skimage
from .._shared import utils
from ..measure import label
from ._inpaint import _build_matrix_inner


def _get_neighborhood(nd_idx, radius, nd_shape):
    bounds_lo = np.maximum(nd_idx - radius, 0)
    bounds_hi = np.minimum(nd_idx + radius + 1, nd_shape)
    return bounds_lo, bounds_hi


def _get_neigh_coef(shape, center, dtype=float):
    # Create biharmonic coefficients ndarray
    neigh_coef = np.zeros(shape, dtype=dtype)
    neigh_coef[center] = 1
    neigh_coef = laplace(laplace(neigh_coef))

    # extract non-zero locations and values
    coef_idx = np.where(neigh_coef)
    coef_vals = neigh_coef[coef_idx]

    coef_idx = np.stack(coef_idx, axis=0)
    return neigh_coef, coef_idx, coef_vals


def _inpaint_biharmonic_single_region(
    image, mask, out, neigh_coef_full, coef_vals, raveled_offsets
):
    """Solve a (sparse) linear system corresponding to biharmonic inpainting.

    This function creates a linear system of the form:

    ``A @ u = b``

    where ``A`` is a sparse matrix, ``b`` is a vector enforcing smoothness and
    boundary constraints and ``u`` is the vector of inpainted values to be
    (uniquely) determined by solving the linear system.

    ``A`` is a sparse matrix of shape (n_mask, n_mask) where ``n_mask``
    corresponds to the number of non-zero values in ``mask`` (i.e. the number
    of pixels to be inpainted). Each row in A will have a number of non-zero
    values equal to the number of non-zero values in the biharmonic kernel,
    ``neigh_coef_full``. In practice, biharmonic kernels with reduced extent
    are used at the image borders. This matrix, ``A`` is the same for all
    image channels (since the same inpainting mask is currently used for all
    channels).

    ``u`` is a dense matrix of shape ``(n_mask, n_channels)`` and represents
    the vector of unknown values for each channel.

    ``b`` is a dense matrix of shape ``(n_mask, n_channels)`` and represents
    the desired output of convolving the solution with the biharmonic kernel.
    At mask locations where there is no overlap with known values, ``b`` will
    have a value of 0. This enforces the biharmonic smoothness constraint in
    the interior of inpainting regions. For regions near the boundary that
    overlap with known values, the entries in ``b`` enforce boundary conditions
    designed to avoid discontinuity with the known values.
    """

    n_channels = out.shape[-1]
    radius = neigh_coef_full.shape[0] // 2

    edge_mask = np.ones(mask.shape, dtype=bool)
    edge_mask[(slice(radius, -radius),) * mask.ndim] = 0
    boundary_mask = edge_mask * mask
    center_mask = ~edge_mask * mask

    boundary_pts = np.where(boundary_mask)
    boundary_i = np.flatnonzero(boundary_mask)
    center_i = np.flatnonzero(center_mask)
    mask_i = np.concatenate((boundary_i, center_i))

    center_pts = np.where(center_mask)
    mask_pts = tuple([np.concatenate((b, c)) for b, c in zip(boundary_pts, center_pts)])

    # Use convolution to predetermine the number of non-zero entries in the
    # sparse system matrix.
    structure = neigh_coef_full != 0
    tmp = ndi.convolve(mask, structure, output=np.uint8, mode='constant')
    nnz_matrix = tmp[mask].sum()

    # Need to estimate the number of zeros for the right hand side vector.
    # The computation below will slightly overestimate the true number of zeros
    # due to edge effects (the kernel itself gets shrunk in size near the
    # edges, but that isn't accounted for here). We can trim any excess entries
    # later.
    n_mask = np.count_nonzero(mask)
    n_struct = np.count_nonzero(structure)
    nnz_rhs_vector_max = n_mask - np.count_nonzero(tmp == n_struct)

    # pre-allocate arrays storing sparse matrix indices and values
    row_idx_known = np.empty(nnz_rhs_vector_max, dtype=np.intp)
    data_known = np.zeros((nnz_rhs_vector_max, n_channels), dtype=out.dtype)
    row_idx_unknown = np.empty(nnz_matrix, dtype=np.intp)
    col_idx_unknown = np.empty(nnz_matrix, dtype=np.intp)
    data_unknown = np.empty(nnz_matrix, dtype=out.dtype)

    # cache the various small, non-square Laplacians used near the boundary
    coef_cache = {}

    # Iterate over masked points near the boundary
    mask_flat = mask.reshape(-1)
    out_flat = np.ascontiguousarray(out.reshape((-1, n_channels)))
    idx_known = 0
    idx_unknown = 0
    mask_pt_n = -1
    boundary_pts = np.stack(boundary_pts, axis=1)
    for mask_pt_n, nd_idx in enumerate(boundary_pts):
        # Get bounded neighborhood of selected radius
        b_lo, b_hi = _get_neighborhood(nd_idx, radius, mask.shape)

        # Create (truncated) biharmonic coefficients ndarray
        coef_shape = tuple(b_hi - b_lo)
        coef_center = tuple(nd_idx - b_lo)
        coef_idx, coefs = coef_cache.get((coef_shape, coef_center), (None, None))
        if coef_idx is None:
            _, coef_idx, coefs = _get_neigh_coef(
                coef_shape, coef_center, dtype=out.dtype
            )
            coef_cache[(coef_shape, coef_center)] = (coef_idx, coefs)

        # compute corresponding 1d indices into the mask
        coef_idx = coef_idx + b_lo[:, np.newaxis]
        index1d = np.ravel_multi_index(coef_idx, mask.shape)

        # Iterate over masked point's neighborhood
        nvals = 0
        for coef, i in zip(coefs, index1d):
            if mask_flat[i]:
                row_idx_unknown[idx_unknown] = mask_pt_n
                col_idx_unknown[idx_unknown] = i
                data_unknown[idx_unknown] = coef
                idx_unknown += 1
            else:
                data_known[idx_known, :] -= coef * out_flat[i, :]
                nvals += 1
        if nvals:
            row_idx_known[idx_known] = mask_pt_n
            idx_known += 1

    # Call an efficient Cython-based implementation for all interior points
    row_start = mask_pt_n + 1
    known_start_idx = idx_known
    unknown_start_idx = idx_unknown
    nnz_rhs = _build_matrix_inner(
        # starting indices
        row_start,
        known_start_idx,
        unknown_start_idx,
        # input arrays
        center_i,
        raveled_offsets,
        coef_vals,
        mask_flat,
        out_flat,
        # output arrays
        row_idx_known,
        data_known,
        row_idx_unknown,
        col_idx_unknown,
        data_unknown,
    )

    # trim RHS vector values and indices to the exact length
    row_idx_known = row_idx_known[:nnz_rhs]
    data_known = data_known[:nnz_rhs, :]

    # Form sparse matrix of unknown values
    sp_shape = (n_mask, out.size)
    matrix_unknown = sparse.csr_array(
        (data_unknown, (row_idx_unknown, col_idx_unknown)), shape=sp_shape
    )

    # Solve linear system for masked points
    matrix_unknown = matrix_unknown[:, mask_i]

    # dense vectors representing the right hand side for each channel
    rhs = np.zeros((n_mask, n_channels), dtype=out.dtype)
    rhs[row_idx_known, :] = data_known

    # set use_umfpack to False so float32 data is supported
    result = spsolve(matrix_unknown, rhs, use_umfpack=False, permc_spec='MMD_ATA')
    if result.ndim == 1:
        result = result[:, np.newaxis]

    out[mask_pts] = result
    return out


@utils.channel_as_last_axis()
def inpaint_biharmonic(image, mask, *, split_into_regions=False, channel_axis=None):
    """Inpaint masked points in image with biharmonic equations.

    Parameters
    ----------
    image : (M[, N[, ..., P]][, C]) ndarray
        Input image.
    mask : (M[, N[, ..., P]]) ndarray
        Array of pixels to be inpainted. Have to be the same shape as one
        of the 'image' channels. Unknown pixels have to be represented with 1,
        known pixels - with 0.
    split_into_regions : boolean, optional
        If True, inpainting is performed on a region-by-region basis. This is
        likely to be slower, but will have reduced memory requirements.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (M[, N[, ..., P]][, C]) ndarray
        Input image with masked pixels inpainted.

    References
    ----------
    .. [1]  S.B.Damelin and N.S.Hoang. "On Surface Completion and Image
            Inpainting by Biharmonic Functions: Numerical Aspects",
            International Journal of Mathematics and Mathematical Sciences,
            Vol. 2018, Article ID 3950312
            :DOI:`10.1155/2018/3950312`
    .. [2]  C. K. Chui and H. N. Mhaskar, MRA Contextual-Recovery Extension of
            Smooth Functions on Manifolds, Appl. and Comp. Harmonic Anal.,
            28 (2010), 104-113,
            :DOI:`10.1016/j.acha.2009.04.004`

    Examples
    --------
    >>> img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    >>> mask = np.zeros_like(img)
    >>> mask[2, 2:] = 1
    >>> mask[1, 3:] = 1
    >>> mask[0, 4:] = 1
    >>> out = inpaint_biharmonic(img, mask)
    """

    if image.ndim < 1:
        raise ValueError('Input array has to be at least 1D')

    multichannel = channel_axis is not None
    img_baseshape = image.shape[:-1] if multichannel else image.shape
    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')

    if np.ma.isMaskedArray(image):
        raise TypeError('Masked arrays are not supported')

    image = skimage.img_as_float(image)

    # float16->float32 and float128->float64
    float_dtype = utils._supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    mask = mask.astype(bool, copy=False)
    if not multichannel:
        image = image[..., np.newaxis]
    out = np.copy(image, order='C')

    # Create biharmonic coefficients ndarray
    radius = 2
    coef_shape = (2 * radius + 1,) * mask.ndim
    coef_center = (radius,) * mask.ndim
    neigh_coef_full, coef_idx, coef_vals = _get_neigh_coef(
        coef_shape, coef_center, dtype=out.dtype
    )

    # stride for the last spatial dimension
    channel_stride_bytes = out.strides[-2]

    # offsets to all neighboring non-zero elements in the footprint
    offsets = coef_idx - radius

    # determine per-channel intensity limits
    known_points = image[~mask]
    limits = (known_points.min(axis=0), known_points.max(axis=0))

    if split_into_regions:
        # Split inpainting mask into independent regions
        kernel = ndi.generate_binary_structure(mask.ndim, 1)
        mask_dilated = ndi.binary_dilation(mask, structure=kernel)
        mask_labeled = label(mask_dilated)
        mask_labeled *= mask

        bbox_slices = ndi.find_objects(mask_labeled)

        for idx_region, bb_slice in enumerate(bbox_slices, 1):
            # expand object bounding boxes by the biharmonic kernel radius
            roi_sl = tuple(
                slice(max(sl.start - radius, 0), min(sl.stop + radius, size))
                for sl, size in zip(bb_slice, mask_labeled.shape)
            )
            # extract only the region surrounding the label of interest
            mask_region = mask_labeled[roi_sl] == idx_region
            # add slice for axes
            roi_sl += (slice(None),)
            # copy for contiguity and to account for possible ROI overlap
            otmp = out[roi_sl].copy()

            # compute raveled offsets for the ROI
            ostrides = np.array(
                [s // channel_stride_bytes for s in otmp[..., 0].strides]
            )
            raveled_offsets = np.sum(offsets * ostrides[..., np.newaxis], axis=0)

            _inpaint_biharmonic_single_region(
                image[roi_sl],
                mask_region,
                otmp,
                neigh_coef_full,
                coef_vals,
                raveled_offsets,
            )
            # assign output to the
            out[roi_sl] = otmp
    else:
        # compute raveled offsets for output image
        ostrides = np.array([s // channel_stride_bytes for s in out[..., 0].strides])
        raveled_offsets = np.sum(offsets * ostrides[..., np.newaxis], axis=0)

        _inpaint_biharmonic_single_region(
            image, mask, out, neigh_coef_full, coef_vals, raveled_offsets
        )

    # Handle enormous values on a per-channel basis
    np.clip(out, a_min=limits[0], a_max=limits[1], out=out)

    if not multichannel:
        out = out[..., 0]

    return out


@utils.channel_as_last_axis()
def inpaint_texture(
    image,
    mask,
    *,
    source_mask=None,
    patch_size=9,
    search_radius=None,
    channel_axis=None,
):
    """Inpaint masked points using exemplars from intact parts of the image.

    This function fills the masked region with small patches sampled from
    intact parts of the image.  Unlike :func:`inpaint_biharmonic`, which favors
    smooth interpolation, this method is intended for structured or periodic
    textures such as wood grain, brickwork, and painted lines.  The fill order
    favors boundary pixels with strong structure, so visible lines and texture
    can continue across the masked region.

    Candidate source patches are never taken from the masked region.  If
    ``source_mask`` is given, candidate patches must be entirely contained in
    the allowed part of ``source_mask``.  Pixels outside ``mask`` are copied
    unchanged from the input, and the input array is never modified.

    Parameters
    ----------
    image : (M, N[, C]) ndarray
        Input grayscale or RGB image.
    mask : (M, N) ndarray
        Array of pixels to be inpainted.  Unknown pixels must be nonzero and
        known pixels zero.
    source_mask : (M, N) ndarray, optional
        Selection of pixels that may be used as source material.  ``True``
        marks allowed pixels.  Pixels in ``mask`` are always excluded from the
        source, even when they are ``True`` in ``source_mask``.  By default,
        all pixels outside ``mask`` may be used.
    patch_size : int, optional
        Side length of the square texture patches.  Must be an odd integer
        greater than or equal to 3.  Larger patches preserve larger-scale
        structure, but require a correspondingly larger source area.
    search_radius : int, optional
        Maximum distance, in pixels, between a fill pixel and a candidate
        source patch center.  By default, source patches from the whole allowed
        region are considered.
    channel_axis : int or None, optional
        If None, the image is assumed to be grayscale.  Otherwise, this
        parameter indicates which axis of the array corresponds to channels.

    Returns
    -------
    out : (M, N[, C]) ndarray
        Input image with masked pixels inpainted.  The output has the same
        dtype as ``image``.

    Raises
    -------
    ValueError
        If the image or masks have incompatible shapes, if the patch size or
        search radius is invalid, or if no complete source patch is available
        in the allowed region.

    Notes
    -----
    The algorithm follows the exemplar-based inpainting strategy of
    Criminisi et al. [1]_.  The highest-priority boundary pixel is chosen
    using a confidence term and an isophote term.  Its neighborhood is
    compared against every permitted source patch, and only the missing
    pixels in that neighborhood are filled.  Since many overlapping patches
    are used, the result is not a single copied rectangle or a uniform fill.

    References
    ----------
    .. [1] A. Criminisi, P. Perez, and K. Toyama, "Region Filling and Object
           Removal by Exemplar-Based Image Inpainting", IEEE Transactions on
           Image Processing, 13(9), 1200-1212, 2004.
           :DOI:`10.1109/TIP.2004.834113`

    Examples
    --------
    >>> img = np.zeros((15, 25))
    >>> img[:, 1::3] = 1
    >>> img[7:9, :] = 0.2
    >>> mask = np.zeros_like(img, dtype=bool)
    >>> mask[6:12, 10:16] = True
    >>> out = inpaint_texture(img, mask, patch_size=9)
    >>> out.shape == img.shape
    True
    >>> np.array_equal(out[~mask], img[~mask])
    True
    """
    if np.ma.isMaskedArray(image):
        raise TypeError('Masked arrays are not supported')
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError('Input image must be a 2D grayscale or RGB image')
    if image.size == 0:
        raise ValueError('Input image must not be empty')

    multichannel = image.ndim == 3
    base_shape = image.shape[:-1] if multichannel else image.shape
    if multichannel and image.shape[-1] == 0:
        raise ValueError('Input image must have at least one channel')

    mask = np.asarray(mask)
    if mask.shape != base_shape:
        raise ValueError('mask must have the same shape as the image')
    mask = mask.astype(bool, copy=False)

    if source_mask is not None:
        source_mask = np.asarray(source_mask)
        if source_mask.shape != base_shape:
            raise ValueError(
                'source_mask must have the same shape as the image'
            )
        source_mask = source_mask.astype(bool, copy=False)

    if (
        not isinstance(patch_size, (int, np.integer))
        or patch_size < 3
        or patch_size % 2 == 0
    ):
        raise ValueError(
            'patch_size must be an odd integer greater than or equal to 3'
        )
    if search_radius is not None:
        if (
            not isinstance(search_radius, (int, np.integer))
            or search_radius < 0
        ):
            raise ValueError(
                'search_radius must be a nonnegative integer or None'
            )

    out = np.copy(image, order='C')
    if not mask.any():
        return out

    # Work in floating point for distances and gradients, but write sampled
    # pixel values directly to ``out`` so that the input dtype and all
    # unmasked values are preserved exactly.
    work = np.array(image, dtype=np.float64, copy=True)
    if not multichannel:
        work = work[..., np.newaxis]
    n_channels = work.shape[-1]

    if not multichannel or n_channels == 1:
        gray = work[..., 0]
    elif n_channels >= 3:
        gray = work[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    else:
        gray = work.mean(axis=-1)

    allowed_source = ~mask
    if source_mask is not None:
        allowed_source &= source_mask

    radius = patch_size // 2
    full_patch_allowed = (
        ndi.minimum_filter(
            allowed_source.astype(np.uint8),
            size=patch_size,
            mode='constant',
            cval=0,
        )
        == 1
    )
    source_centers = np.argwhere(full_patch_allowed)
    rows, cols = gray.shape
    source_centers = source_centers[
        (source_centers[:, 0] >= radius)
        & (source_centers[:, 0] <= rows - radius - 1)
        & (source_centers[:, 1] >= radius)
        & (source_centers[:, 1] <= cols - radius - 1)
    ]
    if source_centers.size == 0:
        raise ValueError(
            'No complete source patch of size '
            f'{patch_size} x {patch_size} is available in the allowed region'
        )

    source_patches = sliding_window_view(
        work, (patch_size, patch_size), axis=(0, 1)
    )
    filled = ~mask
    pending = np.copy(mask)
    confidence = filled.astype(np.float64)
    structure = ndi.generate_binary_structure(2, 1)

    while pending.any():
        eroded_pending = ndi.binary_erosion(
            pending, structure=structure, border_value=0
        )
        fill_front = pending & ~eroded_pending
        if not fill_front.any():
            fill_front = pending

        confidence_term = ndi.uniform_filter(
            confidence, size=patch_size, mode='constant', cval=0
        )

        filled_float = ndi.gaussian_filter(filled.astype(np.float64), 1.0)
        normal_y, normal_x = np.gradient(filled_float)
        normal_norm = np.hypot(normal_x, normal_y)
        normal_norm[normal_norm == 0] = 1

        gradient_y, gradient_x = np.gradient(
            ndi.gaussian_filter(gray, 1.0)
        )
        data_term = np.abs(
            (-gradient_y * normal_x + gradient_x * normal_y) / normal_norm
        )
        priority = np.where(
            fill_front,
            confidence_term * (data_term + 0.01),
            -np.inf,
        )
        fill_y, fill_x = np.unravel_index(np.argmax(priority), priority.shape)

        patch_y0 = max(0, fill_y - radius)
        patch_y1 = min(rows, fill_y + radius + 1)
        patch_x0 = max(0, fill_x - radius)
        patch_x1 = min(cols, fill_x + radius + 1)
        offset_y = np.arange(patch_y0 - fill_y, patch_y1 - fill_y)
        offset_x = np.arange(patch_x0 - fill_x, patch_x1 - fill_x)

        target_patch = work[patch_y0:patch_y1, patch_x0:patch_x1]
        known_in_target = filled[patch_y0:patch_y1, patch_x0:patch_x1]
        if not known_in_target.any():
            raise ValueError(
                'Cannot inpaint texture: the fill patch has no known context'
            )

        center_distance = np.sum(
            (source_centers - np.array([fill_y, fill_x])) ** 2,
            axis=1,
        )
        if search_radius is None:
            candidate_indices = np.arange(source_centers.shape[0])
        else:
            candidate_indices = np.flatnonzero(
                center_distance <= search_radius**2
            )
        if candidate_indices.size == 0:
            raise ValueError(
                'No allowed source patch is within '
                f'search_radius={search_radius}'
            )

        # Score candidates in chunks to keep memory bounded for large images.
        best_score = np.inf
        best_patch = None
        chunk_size = 2048
        for start in range(0, candidate_indices.size, chunk_size):
            stop = min(start + chunk_size, candidate_indices.size)
            chunk_indices = candidate_indices[start:stop]
            chunk_centers = source_centers[chunk_indices]
            chunk_patches = source_patches[
                chunk_centers[:, 0] - radius,
                chunk_centers[:, 1] - radius,
            ]
            chunk_patches = np.moveaxis(chunk_patches, 1, -1)
            chunk_patches = chunk_patches[:, offset_y + radius, :, :]
            chunk_patches = chunk_patches[:, :, offset_x + radius, :]
            difference = chunk_patches - target_patch[np.newaxis]
            scores = np.sum(difference * difference, axis=(1, 2, 3))
            scores += 1e-9 * center_distance[chunk_indices]
            best_in_chunk = int(np.argmin(scores))
            if scores[best_in_chunk] < best_score:
                best_score = scores[best_in_chunk]
                best_patch = chunk_patches[best_in_chunk]

        if best_patch is None or not np.isfinite(best_score):
            raise ValueError(
                'No allowed source patch could be used to fill the mask'
            )

        pending_in_target = pending[patch_y0:patch_y1, patch_x0:patch_x1]
        local_y, local_x = np.nonzero(pending_in_target)
        source_y = offset_y[local_y] + radius
        source_x = offset_x[local_x] + radius
        sampled = best_patch[source_y, source_x]

        out_patch = out[patch_y0:patch_y1, patch_x0:patch_x1]
        if multichannel:
            out_patch[local_y, local_x] = sampled
        else:
            out_patch[local_y, local_x] = sampled[:, 0]
        target_patch[local_y, local_x] = sampled

        confidence_value = confidence_term[fill_y, fill_x]
        confidence_patch = confidence[patch_y0:patch_y1, patch_x0:patch_x1]
        confidence_patch[pending_in_target] = confidence_value
        filled_patch = filled[patch_y0:patch_y1, patch_x0:patch_x1]
        filled_patch[pending_in_target] = True
        pending_in_target[:] = False

    return out
