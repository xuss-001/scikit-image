"""Robust stitching of overlapping images into a common panorama."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import numbers
import warnings

import numpy as np
from scipy import ndimage as ndi

from ..measure import ransac
from ..util import img_as_float
from ._geometric import (
    AffineTransform,
    EuclideanTransform,
    ProjectiveTransform,
    SimilarityTransform,
)
from ._warps import warp


_TRANSFORM_TYPES = {
    'euclidean': EuclideanTransform,
    'similarity': SimilarityTransform,
    'affine': AffineTransform,
    'projective': ProjectiveTransform,
}

_TRANSFORM_INFO = {
    EuclideanTransform: (2, 1),
    SimilarityTransform: (2, 1),
    AffineTransform: (3, 2),
    ProjectiveTransform: (4, 3),
}


def _as_image_list(images):
    if isinstance(images, np.ndarray):
        raise TypeError(
            "`images` must be a sequence of arrays, not a single array. "
            "Use a list such as [image0, image1, ...]."
        )
    try:
        images = list(images)
    except TypeError as exc:
        raise TypeError("`images` must be an iterable of image arrays.") from exc
    if not images:
        raise ValueError("At least one image is required.")

    arrays = []
    for index, image in enumerate(images):
        array = np.asarray(image)
        if array.ndim not in (2, 3):
            raise ValueError(
                f"Image {index} must be a 2-D grayscale or 3-D RGB array; "
                f"got shape {array.shape}."
            )
        if array.ndim == 3 and array.shape[-1] != 3:
            raise ValueError(
                "Color images must have three channels in the last axis; "
                f"image {index} has shape {array.shape}."
            )
        if array.size == 0:
            raise ValueError(f"Image {index} is empty.")
        arrays.append(array)

    ndim = arrays[0].ndim
    if any(array.ndim != ndim for array in arrays):
        raise ValueError("All images must be either grayscale or RGB.")

    # `img_as_float` may return the input array for floating-point images.
    # Copy here so later exposure corrections can never touch user data.
    return [np.array(img_as_float(image), copy=True) for image in arrays]


def _as_points(points, name='points'):
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    if points.ndim == 1:
        if points.size != 2:
            raise ValueError(f"`{name}` must contain (x, y) coordinate pairs.")
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"`{name}` must have shape (N, 2); got {points.shape}.")
    return points


def _points_from_combined(points):
    if (
        isinstance(points, Sequence)
        and len(points) == 2
        and not isinstance(points, (str, bytes))
    ):
        first = np.asarray(points[0], dtype=float)
        second = np.asarray(points[1], dtype=float)
        if first.shape == second.shape and first.ndim == 2 and first.shape[1] == 2:
            return first, second
    points = np.asarray(points, dtype=float)
    if points.ndim == 3 and points.shape[1:] == (2, 2):
        return points[:, 0, :], points[:, 1, :]
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(
            "Combined correspondences must have shape (N, 4) or (N, 2, 2)."
        )
    return points[:, :2], points[:, 2:]


def _validate_pair(i, j, n_images):
    if not isinstance(i, numbers.Integral) or not isinstance(j, numbers.Integral):
        raise ValueError("Image indices in correspondences must be integers.")
    i, j = int(i), int(j)
    if i == j:
        raise ValueError("A correspondence cannot connect an image to itself.")
    if not (0 <= i < n_images and 0 <= j < n_images):
        raise ValueError(
            f"Correspondence image indices ({i}, {j}) are out of range "
            f"for {n_images} images."
        )
    return i, j


def _parse_mapping(correspondences, n_images):
    pairs = []
    for key, value in correspondences.items():
        if isinstance(key, tuple) and len(key) == 2:
            i, j = _validate_pair(*key, n_images)
            if isinstance(value, Mapping):
                if 'src' in value and 'dst' in value:
                    src, dst = value['src'], value['dst']
                elif 'points' in value:
                    src, dst = _points_from_combined(value['points'])
                else:
                    raise ValueError(
                        "A correspondence mapping value must contain 'src' and "
                        "'dst', or 'points'."
                    )
            elif isinstance(value, Sequence) and len(value) == 2:
                src, dst = value
            else:
                src, dst = _points_from_combined(value)
            pairs.append((i, j, src, dst))
        elif isinstance(key, numbers.Integral):
            i = int(key)
            if not 0 <= i < n_images:
                raise ValueError(
                    f"Correspondence image index {i} is out of range for "
                    f"{n_images} images."
                )
            if not isinstance(value, Mapping):
                raise ValueError(
                    "Per-image correspondence mappings must map an image index "
                    "to correspondence data."
                )
            for other, data in value.items():
                if isinstance(other, tuple) and len(other) == 2:
                    j = _validate_pair(*other, n_images)[1]
                else:
                    j = _validate_pair(i, other, n_images)[1]
                if isinstance(data, Mapping):
                    if 'points' in data:
                        src, dst = _points_from_combined(data['points'])
                    else:
                        raise ValueError(
                            "Correspondence mapping values must contain 'points'."
                        )
                elif isinstance(data, Sequence) and len(data) == 2:
                    src, dst = data
                else:
                    src, dst = _points_from_combined(data)
                pairs.append((i, j, src, dst))
        else:
            raise ValueError(
                "Correspondence mappings must use (image_i, image_j) keys."
            )
    return pairs


def _parse_global_points(correspondences, n_images):
    arrays = []
    for points in correspondences:
        arrays.append(_as_points(points))
    lengths = {len(points) for points in arrays}
    if len(lengths) != 1:
        raise ValueError(
            "Global correspondence point arrays must have the same length."
        )
    if len(arrays) != n_images:
        raise ValueError(
            "Global correspondence lists must contain one point array per image."
        )

    pairs = []
    for i in range(n_images):
        for j in range(i + 1, n_images):
            valid = np.all(np.isfinite(arrays[i]), axis=1) & np.all(
                np.isfinite(arrays[j]), axis=1
            )
            pairs.append((i, j, arrays[i][valid], arrays[j][valid]))
    return pairs


def _parse_pair_item(item, n_images, position, n_items):
    if isinstance(item, Mapping):
        nested = _parse_mapping(item, n_images)
        if len(nested) != 1:
            raise ValueError("A correspondence item must describe one image pair.")
        return nested[0]
    if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
        raise ValueError("Invalid correspondence item.")

    if len(item) == 4 and all(
        isinstance(value, numbers.Integral) for value in item[:2]
    ):
        i, j = _validate_pair(item[0], item[1], n_images)
        return i, j, item[2], item[3]
    if len(item) == 3 and all(
        isinstance(value, numbers.Integral) for value in item[:2]
    ):
        i, j = _validate_pair(item[0], item[1], n_images)
        src, dst = _points_from_combined(item[2])
        return i, j, src, dst
    if len(item) == 2:
        if n_images == 2:
            i, j = 0, 1
        elif n_items == n_images - 1:
            i, j = position, position + 1
        else:
            raise ValueError(
                "Ambiguous two-element correspondence. Provide image indices, "
                "or one correspondence per consecutive image pair."
            )
        return i, j, item[0], item[1]
    raise ValueError(
        "Each correspondence must be (i, j, src_points, dst_points), "
        "(i, j, combined_points), or (src_points, dst_points)."
    )


def _parse_correspondences(correspondences, n_images):
    if correspondences is None:
        raise ValueError("`correspondences` must be provided.")

    if isinstance(correspondences, Mapping):
        pairs = _parse_mapping(correspondences, n_images)
    else:
        if isinstance(correspondences, np.ndarray):
            if correspondences.ndim == 2:
                if n_images != 2:
                    raise ValueError(
                        "A single correspondence array can only be used with two "
                        "images. Use (i, j, src_points, dst_points) items otherwise."
                    )
                src, dst = _points_from_combined(correspondences)
                pairs = [(0, 1, src, dst)]
            elif correspondences.ndim == 3 and correspondences.shape[1:] == (2, 2):
                if n_images != 2:
                    raise ValueError(
                        "A single correspondence array can only be used with two "
                        "images."
                    )
                pairs = [(0, 1, correspondences[:, 0], correspondences[:, 1])]
            else:
                raise ValueError("Invalid correspondence array shape.")
        else:
            try:
                items = list(correspondences)
            except TypeError as exc:
                raise TypeError(
                    "`correspondences` must be a mapping or a sequence of image "
                    "point pairs."
                ) from exc
            if not items:
                raise ValueError("`correspondences` is empty.")

            if len(items) == n_images and all(
                isinstance(item, Mapping) for item in items
            ):
                pairs = []
                for item in items:
                    pairs.extend(_parse_mapping(item, n_images))
            elif (
                len(items) == n_images
                and all(
                    not isinstance(item, Mapping)
                    and not (
                        isinstance(item, Sequence)
                        and len(item) in (2, 3, 4)
                        and (
                            len(item) < 2
                            or all(
                                isinstance(value, numbers.Integral)
                                for value in item[:2]
                            )
                        )
                    )
                    for item in items
                )
            ):
                pairs = _parse_global_points(items, n_images)
            elif n_images == 2 and len(items) == 2:
                pairs = [(0, 1, items[0], items[1])]
            else:
                pairs = [
                    _parse_pair_item(item, n_images, index, len(items))
                    for index, item in enumerate(items)
                ]

    cleaned = []
    for i, j, src, dst in pairs:
        src = _as_points(src, 'src_points')
        dst = _as_points(dst, 'dst_points')
        if len(src) != len(dst):
            raise ValueError(
                f"Corresponding point arrays for images ({i}, {j}) have "
                f"different lengths: {len(src)} and {len(dst)}."
            )
        valid = np.all(np.isfinite(src), axis=1) & np.all(np.isfinite(dst), axis=1)
        src, dst = src[valid], dst[valid]
        if len(src) < 2:
            continue
        if i > j:
            i, j, src, dst = j, i, dst, src
        cleaned.append((i, j, src, dst))
    if not cleaned:
        raise ValueError(
            "No valid image correspondences with at least two point pairs were "
            "provided."
        )
    return cleaned


def _is_data_valid(src, dst):
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        return False
    if np.linalg.matrix_rank(np.c_[src, np.ones(len(src))]) < 3:
        return False
    if np.linalg.matrix_rank(np.c_[dst, np.ones(len(dst))]) < 3:
        return False
    return True


def _is_model_valid(model, src, dst):
    if not np.all(np.isfinite(model.params)):
        return False
    matrix = model.params[:2, :2]
    if abs(np.linalg.det(matrix)) <= np.finfo(float).eps:
        return False
    transformed = model(src)
    return np.all(np.isfinite(transformed))


def _fit_transform(src, dst, transform_type, residual_threshold, max_trials, rng):
    min_samples, _ = _TRANSFORM_INFO[transform_type]
    if len(src) < min_samples:
        return None

    if len(src) == min_samples:
        model = transform_type()
        if not model.estimate(src, dst):
            return None
        residuals = model.residuals(src, dst)
        inliers = np.asarray(residuals) <= residual_threshold
        if not np.all(np.isfinite(residuals)) or not np.any(inliers):
            return None
        return model, inliers

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="No inliers found", category=UserWarning
            )
            model, inliers = ransac(
                (src, dst),
                transform_type,
                min_samples=min_samples,
                residual_threshold=residual_threshold,
                max_trials=max_trials,
                is_data_valid=_is_data_valid,
                is_model_valid=_is_model_valid,
                rng=rng,
            )
    except (ValueError, np.linalg.LinAlgError):
        return None
    if model is None or inliers is None or not np.any(inliers):
        return None
    if not np.all(np.isfinite(model.params)):
        return None
    return model, inliers


def _candidate_transform_types(n_points, transform):
    if transform is not None:
        if isinstance(transform, str):
            key = transform.lower()
            if key not in _TRANSFORM_TYPES:
                raise ValueError(
                    f"Unknown transform type {transform!r}. Expected one of "
                    f"{sorted(_TRANSFORM_TYPES)}."
                )
            return [_TRANSFORM_TYPES[key]]
        if transform not in _TRANSFORM_INFO:
            raise ValueError(
                "`transform` must be None, a transform name, or one of "
                "EuclideanTransform, SimilarityTransform, AffineTransform, and "
                "ProjectiveTransform."
            )
        return [transform]

    if n_points >= 5:
        return [ProjectiveTransform, AffineTransform, SimilarityTransform]
    if n_points == 4:
        return [AffineTransform, ProjectiveTransform, SimilarityTransform]
    if n_points == 3:
        return [AffineTransform, SimilarityTransform]
    return [SimilarityTransform, EuclideanTransform]


def _estimate_pair(src, dst, transform, residual_threshold, max_trials, rng):
    for transform_type in _candidate_transform_types(len(src), transform):
        result = _fit_transform(
            src,
            dst,
            transform_type,
            residual_threshold,
            max_trials,
            rng,
        )
        if result is None:
            continue
        model, inliers = result
        min_samples, _ = _TRANSFORM_INFO[transform_type]
        required_inliers = max(min_samples, int(np.ceil(0.5 * len(src))))
        if np.count_nonzero(inliers) >= required_inliers:
            return model, inliers
    return None


def _gray_image(image):
    if image.ndim == 2:
        return image
    return np.mean(image, axis=2)


def _alignment_score(image_src, image_dst, transform_src_to_dst):
    dst_shape = image_dst.shape[:2]
    src_gray = _gray_image(image_src)
    dst_gray = _gray_image(image_dst)
    support = warp(
        np.ones(src_gray.shape, dtype=float),
        transform_src_to_dst.inverse,
        output_shape=dst_shape,
        order=1,
        mode='constant',
        cval=0,
        clip=False,
        preserve_range=True,
    )
    valid = support > 0.95
    overlap = np.count_nonzero(valid) / max(1, min(src_gray.size, dst_gray.size))
    if np.count_nonzero(valid) < 4:
        return -1.0

    warped = warp(
        src_gray,
        transform_src_to_dst.inverse,
        output_shape=dst_shape,
        order=1,
        mode='constant',
        cval=0,
        clip=False,
        preserve_range=True,
    )
    first = warped[valid]
    second = dst_gray[valid]
    if np.std(first) <= np.finfo(float).eps or np.std(second) <= np.finfo(float).eps:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(first, second)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
    return float(overlap * max(correlation, -1.0))


def _oriented_pairs(pairs, point_order):
    if point_order == 'xy':
        return pairs
    if point_order == 'row_col':
        return [
            (i, j, src[:, ::-1], dst[:, ::-1])
            for i, j, src, dst in pairs
        ]
    raise ValueError("`point_order` must be 'auto', 'xy', or 'row_col'.")


def _estimate_edges(
    images,
    pairs,
    transform,
    residual_threshold,
    max_trials,
    rng_seed,
):
    edges = {}
    for edge_index, (i, j, src, dst) in enumerate(pairs):
        result = _estimate_pair(
            src,
            dst,
            transform,
            residual_threshold,
            max_trials,
            rng_seed + edge_index,
        )
        if result is None:
            continue
        model, inliers = result
        if np.count_nonzero(inliers) < 2:
            continue
        quality = _alignment_score(images[i], images[j], model)
        edges[(i, j)] = {
            'transform': model,
            'inliers': inliers,
            'quality': quality,
        }
    return edges


def _select_point_order(
    images,
    pairs,
    point_order,
    transform,
    residual_threshold,
    max_trials,
    rng_seed,
):
    if point_order != 'auto':
        oriented = _oriented_pairs(pairs, point_order)
        return oriented, _estimate_edges(
            images,
            oriented,
            transform,
            residual_threshold,
            max_trials,
            rng_seed,
        )

    candidates = []
    for order in ('xy', 'row_col'):
        oriented = _oriented_pairs(pairs, order)
        edges = _estimate_edges(
            images,
            oriented,
            transform,
            residual_threshold,
            max_trials,
            rng_seed,
        )
        graph = _graph_components(edges, len(images))
        largest = max(len(component) for component in graph)
        quality = sum(edge['quality'] for edge in edges.values())
        candidates.append((largest, quality, order, oriented, edges))

    # Prefer the interpretation that connects more images. Within equally
    # connected interpretations, image-content agreement resolves the order.
    _, _, _, oriented, edges = max(candidates, key=lambda item: item[:2])
    return oriented, edges


def _graph_components(edges, n_images):
    neighbors = defaultdict(set)
    for i, j in edges:
        neighbors[i].add(j)
        neighbors[j].add(i)

    unseen = set(range(n_images))
    components = []
    while unseen:
        start = next(iter(unseen))
        component = set()
        stack = [start]
        unseen.remove(start)
        while stack:
            current = stack.pop()
            component.add(current)
            for neighbor in neighbors[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _global_transforms(edges, n_images, reference):
    neighbors = defaultdict(list)
    for (i, j), edge in edges.items():
        neighbors[i].append((j, False, edge))
        neighbors[j].append((i, True, edge))

    transforms = {reference: ProjectiveTransform()}
    queue = [reference]
    while queue:
        current = queue.pop(0)
        current_to_global = transforms[current]
        for neighbor, reverse, edge in neighbors[current]:
            if neighbor in transforms:
                continue
            edge_transform = edge['transform']
            if reverse:
                neighbor_to_current = edge_transform
            else:
                neighbor_to_current = edge_transform.inverse
            transforms[neighbor] = neighbor_to_current + current_to_global
            queue.append(neighbor)
    return transforms


def _canvas_transform(images, transforms):
    corners = []
    for index, image in enumerate(images):
        height, width = image.shape[:2]
        image_corners = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=float,
        )
        corners.append(transforms[index](image_corners))
    corners = np.concatenate(corners, axis=0)
    if not np.all(np.isfinite(corners)):
        raise ValueError("The estimated transforms produced non-finite coordinates.")

    # Least-squares transform estimation leaves tiny numerical errors around
    # integer-valued image borders. Snap those boundaries before rounding so a
    # pure translation does not grow the canvas by an extra row or column.
    nearest = np.round(corners)
    tolerance = 1e-4 * max(1, max(images[0].shape[:2]))
    close = np.abs(corners - nearest) <= tolerance
    corners[close] = nearest[close]

    min_x = np.floor(np.min(corners[:, 0]))
    min_y = np.floor(np.min(corners[:, 1]))
    max_x = np.ceil(np.max(corners[:, 0])) + 1
    max_y = np.ceil(np.max(corners[:, 1])) + 1
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    if width <= 0 or height <= 0:
        raise ValueError("The estimated panorama has an invalid shape.")

    translation = ProjectiveTransform(
        matrix=np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0, 0, 1]])
    )
    return translation, (height, width)


def _warp_to_canvas(image, transform, output_shape):
    support = warp(
        np.ones(image.shape[:2], dtype=float),
        transform.inverse,
        output_shape=output_shape,
        order=1,
        mode='constant',
        cval=0,
        clip=False,
        preserve_range=True,
    )
    mask = support > 1e-5

    if image.ndim == 3:
        warped = np.empty((*output_shape, image.shape[2]), dtype=float)
        for channel in range(image.shape[2]):
            warped[..., channel] = warp(
                image[..., channel],
                transform.inverse,
                output_shape=output_shape,
                order=1,
                mode='constant',
                cval=0,
                clip=False,
                preserve_range=True,
            )
    else:
        warped = warp(
            image,
            transform.inverse,
            output_shape=output_shape,
            order=1,
            mode='constant',
            cval=0,
            clip=False,
            preserve_range=True,
        )

    # Normalize by the interpolated support to avoid a dark fringe where the
    # bilinear kernel crosses the transformed image boundary.
    if image.ndim == 3:
        warped[mask] /= support[mask, None]
    else:
        warped[mask] /= support[mask]
    return warped, mask


def _fit_affine_intensity(first, second):
    if len(first) > 200_000:
        step = int(np.ceil(len(first) / 200_000))
        first = first[::step]
        second = second[::step]

    if np.var(first) <= np.finfo(float).eps:
        return 1.0, float(np.median(second) - np.median(first))

    slope, intercept = np.polyfit(first, second, 1)
    residuals = second - (slope * first + intercept)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad > np.finfo(float).eps:
        keep = np.abs(residuals - np.median(residuals)) <= 4.5 * mad
        if np.count_nonzero(keep) >= 4 and np.var(first[keep]) > np.finfo(float).eps:
            slope, intercept = np.polyfit(first[keep], second[keep], 1)

    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0.05:
        slope = 1.0
        intercept = float(np.median(second) - np.median(first))
    return float(np.clip(slope, 0.1, 10.0)), float(intercept)


def _overlap_models(warped_images, masks, edges):
    models = {}
    for (i, j) in edges:
        overlap = masks[i] & masks[j]
        if np.count_nonzero(overlap) < 16:
            continue
        first = warped_images[i][overlap]
        second = warped_images[j][overlap]
        if first.ndim == 1:
            slope, intercept = _fit_affine_intensity(first, second)
            models[(i, j)] = (
                np.array([slope]),
                np.array([intercept]),
            )
        else:
            slopes = []
            intercepts = []
            for channel in range(first.shape[1]):
                slope, intercept = _fit_affine_intensity(
                    first[:, channel], second[:, channel]
                )
                slopes.append(slope)
                intercepts.append(intercept)
            models[(i, j)] = (np.asarray(slopes), np.asarray(intercepts))
    return models


def _exposure_corrections(overlap_models, edges, n_images, reference):
    corrections = {reference: (np.ones(1), np.zeros(1))}
    if not overlap_models:
        sample = np.ones(1)
        return {index: (sample.copy(), np.zeros(1)) for index in range(n_images)}

    sample = next(iter(overlap_models.values()))[0]
    corrections[reference] = (np.ones_like(sample), np.zeros_like(sample))
    neighbors = defaultdict(list)
    for (i, j) in edges:
        if (i, j) not in overlap_models:
            continue
        slope, intercept = overlap_models[(i, j)]
        neighbors[i].append((j, slope, intercept))
        neighbors[j].append((i, 1.0 / slope, -intercept / slope))

    queue = [reference]
    while queue:
        current = queue.pop(0)
        current_gain, current_offset = corrections[current]
        for neighbor, slope, intercept in neighbors[current]:
            if neighbor in corrections:
                continue
            gain = current_gain / slope
            offset = current_offset - gain * intercept
            corrections[neighbor] = (gain, offset)
            queue.append(neighbor)

    fallback = (np.ones_like(sample), np.zeros_like(sample))
    return {index: corrections.get(index, fallback) for index in range(n_images)}


def _blend(warped_images, masks, corrections):
    shape = warped_images[0].shape
    accumulator = np.zeros(shape, dtype=float)
    weights_sum = np.zeros(shape[:2], dtype=float)

    ordered_corrections = [corrections[index] for index in range(len(warped_images))]
    for image, mask, (gain, offset) in zip(
        warped_images, masks, ordered_corrections
    ):
        distance = ndi.distance_transform_edt(mask)
        if not np.any(mask):
            continue
        weights = distance / max(1.0, float(np.max(distance[mask])))
        weights[~mask] = 0
        if image.ndim == 3:
            corrected = image * gain.reshape(1, 1, -1) + offset.reshape(1, 1, -1)
            accumulator += corrected * weights[..., None]
        else:
            corrected = image * gain[0] + offset[0]
            accumulator += corrected * weights
        weights_sum += weights

    covered = weights_sum > np.finfo(float).eps
    panorama = np.zeros(shape, dtype=float)
    if panorama.ndim == 3:
        panorama[covered] = accumulator[covered] / weights_sum[covered, None]
    else:
        panorama[covered] = accumulator[covered] / weights_sum[covered]
    return panorama, covered


def stitch_images(
    images,
    correspondences=None,
    *,
    reference=0,
    transform=None,
    residual_threshold=2.0,
    max_trials=1000,
    point_order='auto',
    exposure_compensation=True,
    rng=None,
    return_mask=True,
    return_alpha=False,
    matches=None,
    points=None,
):
    """Stitch overlapping images into a single panorama.

    Images are connected by point correspondences. Only pairwise overlap is
    required: images may be linked indirectly through other images. Robust
    model fitting (RANSAC) rejects a small number of incorrect point pairs.

    Parameters
    ----------
    images : sequence of ndarray
        Two or more grayscale ``(M, N)`` or RGB ``(M, N, 3)`` images. Images
        may have different sizes. Input arrays are never modified.
    correspondences : mapping or sequence
        Point correspondences. Each pair may be specified as
        ``(image_i, image_j, points_i, points_j)``, where both point arrays
        have shape ``(P, 2)``. A combined ``(P, 4)`` array
        ``[x_i, y_i, x_j, y_j]`` may be used as the third item instead.
        Mappings with ``(image_i, image_j)`` keys and either
        ``(points_i, points_j)`` or combined point arrays as values are also
        accepted. For exactly two images, a single combined array or a
        ``(points_i, points_j)`` pair is accepted.

        If one point array per image is supplied, row ``k`` is interpreted as
        the same physical point in every image. Otherwise, when one
        ``(points_i, points_j)`` item is supplied per consecutive image pair,
        those pairs are used in order.
    reference : int, optional
        Index of the image used as the global coordinate frame. The final
        canvas is translated to include all images, so this does not crop the
        result.
    transform : {None, str, class}, optional
        Geometric model used for each image pair. ``None`` selects a robust
        model automatically according to the number of available points:
        projective for five or more points, affine for three or four, and
        similarity for two. A transform class such as
        :class:`~skimage.transform.ProjectiveTransform` or one of the strings
        ``'euclidean'``, ``'similarity'``, ``'affine'``, and ``'projective'``
        forces a particular model.
    residual_threshold : float, optional
        Maximum RANSAC inlier residual, in pixels.
    max_trials : int, optional
        Maximum number of RANSAC iterations per pair.
    point_order : {'auto', 'xy', 'row_col'}, optional
        Order of coordinates in the correspondence arrays. scikit-image
        transform coordinates normally use ``(x, y)``. ``'row_col'`` swaps
        them, and ``'auto'`` selects the ordering whose alignments agree best
        with the image content.
    exposure_compensation : bool, optional
        If True, estimate a global intensity gain and offset for every image
        and channel from the overlaps.
    rng : {None, int, numpy.random.Generator}, optional
        Seed or generator for reproducible RANSAC sampling.
    return_mask : bool, optional
        If True (the default), return ``(panorama, mask)``. If False, return
        only the panorama.
    return_alpha : bool, optional
        If True, append a coverage alpha channel to the returned panorama.
        This is supported for both grayscale and RGB inputs.
    matches, points : optional
        Aliases for ``correspondences``.

    Returns
    -------
    panorama : ndarray
        Floating-point composite with the same number of spatial dimensions
        as the inputs. Values are in the same normalized range as
        :func:`skimage.util.img_as_float`. Uncovered pixels are zero.
    mask : ndarray of bool
        Coverage mask with shape ``panorama.shape[:2]``. It is returned only
        when ``return_mask`` is True. Use it as an alpha channel or
        ``np.where`` condition for transparent display.

    Raises
    ------
    ValueError
        If correspondences are missing or invalid, if an image cannot be
        connected to the rest of the group, or if the estimated geometry is
        degenerate.

    Notes
    -----
    The pairwise transformation maps coordinates in image ``i`` to image
    ``j``. Correspondences should describe the same pixels under that
    convention. A small number of grossly incorrect matches is handled by
    RANSAC; at least five point pairs per image pair are recommended when a
    projective model is desired.

    Examples
    --------
    Stitch three horizontally shifted crops using pairwise correspondences:

    >>> import numpy as np
    >>> from skimage.transform import stitch_images
    >>> panorama, mask = stitch_images(
    ...     images, [(0, 1, points_01_a, points_01_b),
    ...              (1, 2, points_12_a, points_12_b)]
    ... )  # doctest: +SKIP
    """
    if correspondences is None:
        correspondences = matches if matches is not None else points
    elif matches is not None or points is not None:
        raise ValueError(
            "Pass correspondences only once, using `correspondences`, `matches`, "
            "or `points`."
        )
    if correspondences is None:
        raise ValueError(
            "`correspondences` must be provided as pairwise image point matches."
        )

    images = _as_image_list(images)
    n_images = len(images)
    if not 0 <= reference < n_images:
        raise ValueError(
            f"`reference` must be between 0 and {n_images - 1}, got {reference}."
        )
    if residual_threshold <= 0:
        raise ValueError("`residual_threshold` must be positive.")
    if max_trials < 1:
        raise ValueError("`max_trials` must be at least one.")
    if point_order not in ('auto', 'xy', 'row_col'):
        raise ValueError("`point_order` must be 'auto', 'xy', or 'row_col'.")

    pairs = _parse_correspondences(correspondences, n_images)
    seed = np.random.default_rng(rng)
    rng_seed = int(seed.integers(0, np.iinfo(np.int32).max))
    pairs, edges = _select_point_order(
        images,
        pairs,
        point_order,
        transform,
        residual_threshold,
        max_trials,
        rng_seed,
    )
    if not edges:
        raise ValueError(
            "No image pair could be aligned from the supplied correspondences. "
            "Check that point pairs match the requested coordinate order and "
            "that each pair has enough inliers."
        )

    components = _graph_components(edges, n_images)
    if len(components) != 1:
        components = sorted(
            (sorted(component) for component in components),
            key=lambda component: (-len(component), component),
        )
        details = "; ".join(
            f"component {index}: images {component}"
            for index, component in enumerate(components)
        )
        raise ValueError(
            "The images cannot be connected into one panorama. "
            f"Disconnected image groups are: {details}. Add at least two "
            "corresponding points between images that bridge these groups."
        )

    transforms = _global_transforms(edges, n_images, reference)
    canvas_transform, output_shape = _canvas_transform(images, transforms)

    warped_images = []
    masks = []
    for index, image in enumerate(images):
        image_to_canvas = transforms[index] + canvas_transform
        warped, mask = _warp_to_canvas(image, image_to_canvas, output_shape)
        warped_images.append(warped)
        masks.append(mask)

    if exposure_compensation:
        corrections = _exposure_corrections(
            _overlap_models(warped_images, masks, edges),
            edges,
            n_images,
            reference,
        )
    else:
        sample = np.ones(1) if images[0].ndim == 2 else np.ones(images[0].shape[2])
        corrections = {
            index: (sample.copy(), np.zeros_like(sample))
            for index in range(n_images)
        }
    panorama, coverage = _blend(warped_images, masks, corrections)

    finite_values = np.concatenate(
        [image[np.isfinite(image)].ravel() for image in images]
    )
    if finite_values.size:
        low, high = float(np.min(finite_values)), float(np.max(finite_values))
        if high > low:
            panorama = np.clip(panorama, low, high)
    panorama[~coverage] = 0

    if return_alpha:
        alpha = coverage.astype(float)
        panorama = np.dstack([panorama, alpha])
    if return_mask:
        return panorama, coverage
    return panorama


# Common spellings for the same high-level operation. The canonical name is
# ``stitch_images``; these aliases keep the API approachable without requiring
# callers to know whether stitching belongs under registration or transform.
stitch = stitch_images
panorama = stitch_images
panorama_stitch = stitch_images


__all__ = ['stitch_images', 'stitch', 'panorama', 'panorama_stitch']
