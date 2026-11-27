import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from skimage import registration, transform
from skimage.transform import stitch_images


def _three_overlapping_crops(*, rgb=False, exposure=(1.0, 1.0, 1.0)):
    rng = np.random.default_rng(42)
    shape = (100, 240) if not rgb else (100, 240, 3)
    scene = rng.random(shape)

    offsets = [(0, 0), (8, 55), (15, 135)]
    sizes = [(80, 100), (78, 95), (82, 100)]
    images = []
    for (row, col), (height, width), gain in zip(offsets, sizes, exposure):
        images.append(scene[row : row + height, col : col + width].copy() * gain)

    def local_points(global_points, index):
        row, col = offsets[index]
        return global_points - np.array([col, row])

    points_01 = np.array(
        [
            [60, 10],
            [70, 20],
            [85, 35],
            [95, 55],
            [65, 60],
            [80, 45],
            [90, 15],
            [75, 8],
        ],
        dtype=float,
    )
    points_12 = np.array(
        [
            [138, 20],
            [146, 30],
            [142, 50],
            [147, 65],
            [136, 45],
            [144, 12],
            [140, 70],
            [145, 40],
        ],
        dtype=float,
    )
    correspondences = [
        (
            0,
            1,
            local_points(points_01, 0),
            local_points(points_01, 1),
        ),
        (
            1,
            2,
            local_points(points_12, 1),
            local_points(points_12, 2),
        ),
    ]
    return scene, images, correspondences


def test_stitch_three_images_via_indirect_connection():
    scene, images, correspondences = _three_overlapping_crops()
    originals = [image.copy() for image in images]
    correspondence_copies = [
        (i, j, src.copy(), dst.copy()) for i, j, src, dst in correspondences
    ]

    panorama, mask = stitch_images(
        images, correspondences, point_order='xy', rng=0
    )

    assert panorama.ndim == 2
    assert mask.shape == panorama.shape
    assert mask.dtype == bool
    assert np.count_nonzero(mask) > 0.75 * mask.size
    expected = scene[: panorama.shape[0], : panorama.shape[1]]
    assert np.mean((panorama[mask] - expected[mask]) ** 2) < 1e-3

    # Image 2 is only linked to image 0 through image 1.
    assert mask[20, 200]
    assert not mask[95, 10]

    for image, original in zip(images, originals):
        assert_array_equal(image, original)
    for (_, _, src, dst), (_, _, src0, dst0) in zip(
        correspondences, correspondence_copies
    ):
        assert_array_equal(src, src0)
        assert_array_equal(dst, dst0)


def test_stitch_rejects_bad_correspondences():
    _, images, correspondences = _three_overlapping_crops()
    first = correspondences[0]
    bad_dst = first[3].copy()
    bad_dst[2] += [180, -90]
    correspondences[0] = (first[0], first[1], first[2], bad_dst)

    panorama, mask = stitch_images(
        images,
        correspondences,
        point_order='xy',
        rng=0,
    )
    expected = np.zeros_like(panorama)
    scene, _, _ = _three_overlapping_crops()
    expected[...] = scene[: panorama.shape[0], : panorama.shape[1]]
    assert np.mean((panorama[mask] - expected[mask]) ** 2) < 1e-3


def test_stitch_balances_exposure_and_removes_seam():
    scene, images, correspondences = _three_overlapping_crops(
        exposure=(1.0, 1.45, 0.65)
    )
    panorama, mask = stitch_images(
        images,
        correspondences,
        point_order='xy',
        rng=0,
    )

    corrected = np.mean((panorama[mask] - scene[: panorama.shape[0], : panorama.shape[1]][mask]) ** 2)
    assert corrected < 5e-3

    no_exposure, no_exposure_mask = stitch_images(
        images,
        correspondences,
        point_order='xy',
        exposure_compensation=False,
        rng=0,
    )
    uncorrected = np.mean(
        (
            no_exposure[no_exposure_mask]
            - scene[: no_exposure.shape[0], : no_exposure.shape[1]][
                no_exposure_mask
            ]
        )
        ** 2
    )
    assert corrected < 0.1 * uncorrected

    # The overlap at each seam should change gradually rather than showing a
    # discontinuous brightness step.
    strip = panorama[15:55, 70:95]
    expected_strip = scene[15:55, 70:95]
    seam_steps = np.diff(np.mean(strip, axis=0))
    expected_steps = np.diff(np.mean(expected_strip, axis=0))
    assert np.max(np.abs(seam_steps - expected_steps)) < 0.02


def test_stitch_rgb_and_alpha_output():
    _, images, correspondences = _three_overlapping_crops(rgb=True)
    panorama, mask = stitch_images(
        images,
        correspondences,
        point_order='xy',
        rng=0,
        return_alpha=True,
    )

    assert panorama.shape[:2] == mask.shape
    assert panorama.shape[2] == 4
    assert_allclose(panorama[..., 3], mask)
    assert np.all(panorama[~mask, :3] == 0)


def test_stitch_balances_additive_exposure_shift():
    scene, images, correspondences = _three_overlapping_crops()
    images = [images[0], images[1] + 0.03, images[2] - 0.04]
    panorama, mask = stitch_images(
        images,
        correspondences,
        point_order='xy',
        rng=0,
    )
    expected = scene[: panorama.shape[0], : panorama.shape[1]]
    assert np.mean((panorama[mask] - expected[mask]) ** 2) < 1e-3


def test_stitch_row_col_auto_detection():
    _, images, correspondences = _three_overlapping_crops()
    row_col = [
        (i, j, src[:, ::-1], dst[:, ::-1])
        for i, j, src, dst in correspondences
    ]

    panorama, mask = stitch_images(images, row_col, rng=0)
    scene, _, _ = _three_overlapping_crops()
    expected = scene[: panorama.shape[0], : panorama.shape[1]]
    assert np.mean((panorama[mask] - expected[mask]) ** 2) < 1e-3


def test_stitch_accepts_mapping_and_combined_points():
    _, images, correspondences = _three_overlapping_crops()
    combined = {}
    for i, j, src, dst in correspondences:
        combined[(i, j)] = np.c_[src, dst]

    panorama, mask = stitch_images(
        images,
        combined,
        point_order='xy',
        rng=0,
    )
    assert np.any(mask)

    only_panorama = stitch_images(
        images,
        combined,
        point_order='xy',
        rng=0,
        return_mask=False,
    )
    assert_array_equal(only_panorama, panorama)


def test_stitch_rejects_disconnected_images():
    _, images, correspondences = _three_overlapping_crops()
    with pytest.raises(ValueError, match="cannot be connected"):
        stitch_images(
            images,
            correspondences[:1],
            point_order='xy',
            rng=0,
        )


def test_stitch_is_available_from_registration():
    assert registration.stitch_images is stitch_images
    assert transform.panorama is stitch_images
