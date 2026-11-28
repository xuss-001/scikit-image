import pytest
import numpy as np
from skimage import data
from skimage.measure._label import _label_bool, label, label_chunked
from skimage.measure._ccomp import label_cython as clabel

from skimage._shared import testing

# In this testsuite, we ensure that the results provided by
# label_cython are identical to the one from _label_bool,
# which is based on ndimage.


def test_no_option():
    img = data.binary_blobs(length=128, blob_size_fraction=0.15, n_dim=3)
    l_ndi = _label_bool(img)
    l_cy = clabel(img)
    testing.assert_equal(l_ndi, l_cy)


def test_background():
    img = data.binary_blobs(length=128, blob_size_fraction=0.15, n_dim=3)
    l_ndi = _label_bool(img, background=0)
    l_cy = clabel(img, background=0)
    testing.assert_equal(l_ndi, l_cy)

    l_ndi = _label_bool(img, background=1)
    l_cy = clabel(img, background=1)
    testing.assert_equal(l_ndi, l_cy)


def test_return_num():
    img = data.binary_blobs(length=128, blob_size_fraction=0.15, n_dim=3)
    l_ndi = _label_bool(img, return_num=True)
    l_cy = clabel(img, return_num=True)
    testing.assert_equal(l_ndi, l_cy)


def test_connectivity():
    img = data.binary_blobs(length=128, blob_size_fraction=0.15, n_dim=3)
    for c in (1, 2, 3):
        l_ndi = _label_bool(img, connectivity=c)
        l_cy = clabel(img, connectivity=c)
        testing.assert_equal(l_ndi, l_cy)

    for c in (0, 4):
        with pytest.raises(ValueError):
            l_ndi = _label_bool(img, connectivity=c)
        with pytest.raises(ValueError):
            l_cy = clabel(img, connectivity=c)


@pytest.mark.parametrize("dtype", [bool, int])
def test_zero_size(dtype):
    img = np.ones((300, 0, 300), dtype=dtype)
    lab, num = label(img, return_num=True)

    assert lab.shape == img.shape
    assert num == 0


@pytest.mark.parametrize(
    "first, second, first_value, second_value, connectivity, connected",
    [
        pytest.param(
            (0, 0, 1), (0, 0, 2), 1, 1, 1, True, id="face"
        ),
        pytest.param(
            (0, 1, 1), (0, 0, 2), 1, 1, 2, True, id="face-tangent"
        ),
        pytest.param(
            (0, 1, 1), (0, 2, 2), 1, 1, 2, True, id="edge"
        ),
        pytest.param(
            (1, 1, 1), (2, 2, 2), 1, 1, 3, True, id="corner"
        ),
        pytest.param(
            (1, 1, 1), (2, 2, 2), 1, 1, 2, False, id="corner-no-connectivity"
        ),
        pytest.param(
            (0, 0, 1), (0, 0, 2), 1, 2, 3, False, id="different-materials"
        ),
    ],
)
def test_chunked_connections_across_chunk_boundaries(
    first, second, first_value, second_value, connectivity, connected
):
    image = np.zeros((4, 4, 4), dtype=np.int16)
    image[first] = first_value
    image[second] = second_value

    expected = label(image, connectivity=connectivity)
    assert bool(expected[first] == expected[second]) is connected

    for chunks in [(2, 2, 2), (1, 3, 2), (4, 4, 4)]:
        labels = label_chunked(
            image, chunks=chunks, connectivity=connectivity
        )
        testing.assert_equal(labels, expected)


@pytest.mark.parametrize("connectivity", [1, 2, 3])
@pytest.mark.parametrize("background", [None, 0, -1])
def test_chunked_matches_full_label(connectivity, background):
    rng = np.random.default_rng(1234)
    image = rng.integers(-1, 4, size=(6, 7, 8), dtype=np.int16)
    expected = label(
        image, background=background, connectivity=connectivity
    )

    for chunks in [(1, 3, 2), (2, 2, 2), (4, 5, 3), (20, 20, 20)]:
        actual, num = label_chunked(
            image,
            chunks=chunks,
            background=background,
            connectivity=connectivity,
            return_num=True,
        )
        testing.assert_equal(actual, expected)
        assert num == expected.max()


def test_chunked_memmap_output(tmp_path):
    image = np.zeros((6, 6, 6), dtype=np.int16)
    image[1:5, 2:5, 1:4] = 3
    image[2:4, 2:4, 1:3] = 5
    original = image.copy()

    path = tmp_path / "labels.dat"
    out = np.memmap(path, mode="w+", dtype=np.int32, shape=image.shape)
    labels, num = label_chunked(
        image, chunks=(2, 3, 2), out=out, return_num=True
    )
    assert labels is out
    assert num == label(image).max()
    out.flush()
    del out

    expected = label(image)
    reloaded = np.memmap(path, mode="r", dtype=np.int32, shape=image.shape)
    testing.assert_equal(reloaded, expected)
    testing.assert_equal(image, original)


class _SliceGuard:
    def __init__(self, array, max_chunk_size):
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self._max_chunk_size = max_chunk_size
        self.max_seen = 0

    def _check_slices(self, key):
        assert isinstance(key, tuple) and len(key) == self._array.ndim
        assert all(isinstance(index, slice) for index in key)
        size = np.prod([index.stop - index.start for index in key])
        self.max_seen = max(self.max_seen, size)
        assert size <= self._max_chunk_size

    def __getitem__(self, key):
        self._check_slices(key)
        return self._array[key]

    def __setitem__(self, key, value):
        self._check_slices(key)
        self._array[key] = value


def test_chunked_uses_only_block_sized_io():
    rng = np.random.default_rng(5)
    image = rng.integers(0, 3, size=(6, 6, 6), dtype=np.int16)
    guarded_image = _SliceGuard(image, max_chunk_size=2 * 2 * 3)

    out = np.empty(image.shape, dtype=np.int32)
    guarded_out = _SliceGuard(out, max_chunk_size=2 * 2 * 3)
    returned = label_chunked(
        guarded_image, chunks=(2, 2, 3), out=guarded_out
    )

    assert returned is guarded_out
    testing.assert_equal(out, label(image))
    assert guarded_image.max_seen <= 2 * 2 * 3
    assert guarded_out.max_seen <= 2 * 2 * 3
