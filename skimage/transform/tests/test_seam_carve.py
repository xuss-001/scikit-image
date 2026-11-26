import numpy as np
import pytest

from skimage import data
from skimage.transform import seam_carve


def test_seam_carve_grayscale_target_shape():
    image = data.camera()
    carved = seam_carve(image, (400, 450))
    assert carved.shape == (400, 450)


def test_seam_carve_rgb_target_shape():
    image = data.astronaut()
    carved = seam_carve(image, (400, 420))
    assert carved.shape == (400, 420, 3)


def test_seam_carve_preserves_dtype():
    image = data.astronaut()
    assert seam_carve(image, (500, 500)).dtype == image.dtype

    float_image = image.astype(np.float64) / 255.0
    assert seam_carve(float_image, (500, 500)).dtype == np.float64


def test_seam_carve_no_op():
    image = data.camera()
    carved = seam_carve(image, image.shape)
    np.testing.assert_array_equal(carved, image)


def test_seam_carve_does_not_modify_inputs():
    image = data.astronaut()
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[100:300, 200:400] = True
    image_before = image.copy()
    mask_before = mask.copy()

    seam_carve(image, (450, 460), mask=mask)

    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_seam_carve_avoids_low_energy_column():
    # Every row of this constant image has zero energy except the pixels
    # bordering the dark stripe, so the unique minimum-cost seam runs
    # straight through the stripe and removes it first.
    image = np.full((12, 10), 200, dtype=np.uint8)
    image[:, 4] = 0

    carved = seam_carve(image, (12, 9))

    assert carved.shape == (12, 9)
    expected = np.delete(image, 4, axis=1)
    np.testing.assert_array_equal(carved, expected)


def test_seam_carve_protected_region_pixels_are_kept():
    # Encode each column with a unique value so the surviving original
    # columns can be identified unambiguously in the carved result.
    image = np.tile(
        np.arange(20, dtype=np.uint8)[np.newaxis, :], (20, 1)
    )
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:16, 6:14] = True

    carved = seam_carve(image, (20, 12), mask=mask)

    assert carved.shape == (20, 12)
    surviving_columns = np.unique(carved)
    # None of the eight protected columns may be removed.
    np.testing.assert_array_equal(
        np.isin(np.arange(6, 14), surviving_columns), np.ones(8, dtype=bool)
    )
    # Protected pixels keep both their original values and their shape.
    for original_col in range(6, 14):
        new_col = np.searchsorted(
            surviving_columns, original_col
        )
        np.testing.assert_array_equal(
            carved[4:16, new_col], image[4:16, original_col]
        )

    carved_rgb = seam_carve(
        np.repeat(image[..., np.newaxis], 3, axis=2).copy(),
        (20, 12),
        mask=mask,
    )
    assert carved_rgb.shape == (20, 12, 3)


def test_seam_carve_protected_stripe_keeps_shape():
    # A full-height protected stripe cannot lose any of its columns, so the
    # values inside it must remain contiguous and completely unaltered.
    image = data.astronaut()
    stripe = slice(200, 260)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[:, stripe] = True
    original_stripe = image[:, stripe, :].copy()

    carved = seam_carve(image, (512, 462), mask=mask)

    assert carved.shape == (512, 462, 3)
    # The stripe columns cannot be removed, so the full stripe survives
    # contiguously even though its horizontal position may shift.
    assert any(
        np.array_equal(original_stripe, carved[:, start : start + 60, :])
        for start in range(carved.shape[1] - 60 + 1)
    )


def test_seam_carve_protected_rows_kept():
    # Full-width protected rows stay pixel-identical when carving height.
    image = data.astronaut()
    rows = slice(150, 200)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[rows, :] = True
    original_rows = image[rows, :, :].copy()

    carved = seam_carve(image, (462, 512), mask=mask)

    assert carved.shape == (462, 512, 3)
    # The protected rows cannot be removed, so their exact pixel content
    # must survive (their row indices shift as unprotected rows go away).
    for row in original_rows:
        assert any(np.array_equal(row, kept) for kept in carved)


def test_seam_carve_raises_when_protection_makes_target_unreachable():
    image = data.astronaut()
    mask = np.zeros(image.shape[:2], dtype=bool)
    # Protecting 300 columns allows removing at most 212 columns.
    mask[:, 100:400] = True

    with pytest.raises(ValueError, match="protected"):
        seam_carve(image, (512, 100), mask=mask)

    tall_mask = np.zeros(image.shape[:2], dtype=bool)
    tall_mask[100:400, :] = True
    with pytest.raises(ValueError, match="protected"):
        seam_carve(image, (100, 512), mask=tall_mask)


def test_seam_carve_rejects_upscaling():
    image = data.camera()
    with pytest.raises(ValueError, match="only reduces"):
        seam_carve(image, (image.shape[0], image.shape[1] + 1))


def test_seam_carve_rejects_invalid_image():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        seam_carve(rgba, (8, 6))


def test_seam_carve_rejects_invalid_output_shape():
    image = data.camera()
    with pytest.raises(ValueError):
        seam_carve(image, (8,))
    with pytest.raises(ValueError):
        seam_carve(image, (0, 8))
    with pytest.raises(ValueError):
        seam_carve(image, (8.0, 8.0))


def test_seam_carve_rejects_mask_shape_mismatch():
    image = data.camera()
    mask = np.ones((image.shape[0], image.shape[1] - 1), dtype=bool)
    with pytest.raises(ValueError, match="mask"):
        seam_carve(image, (400, 400), mask=mask)
