import itertools

import numpy as np
from scipy import ndimage

from ._ccomp import _get_swaps, label_cython as clabel


def _normalize_chunks(chunks, shape):
    if np.isscalar(chunks):
        chunks = (chunks,) * len(shape)
    else:
        chunks = tuple(chunks)

    if len(chunks) != len(shape):
        raise ValueError(
            f"chunks must have one entry per dimension, got {len(chunks)} "
            f"entries for a {len(shape)}D image."
        )

    normalized = []
    for chunk_size in chunks:
        if not isinstance(chunk_size, (int, np.integer)):
            raise TypeError("chunk sizes must be integers")
        if chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        normalized.append(int(chunk_size))
    return tuple(normalized)


def _chunk_intervals(shape, chunks):
    return [
        [(start, min(start + chunk_size, size)) for start in range(0, size, chunk_size)]
        for size, chunk_size in zip(shape, chunks)
    ]


def _chunk_shape(intervals, chunk_index):
    return tuple(
        intervals[dim][chunk_index[dim]][1] - intervals[dim][chunk_index[dim]][0]
        for dim in range(len(intervals))
    )


def _chunk_slices(intervals, chunk_index):
    return tuple(
        slice(*intervals[dim][chunk_index[dim]]) for dim in range(len(intervals))
    )


def _chunk_strides(n_chunks):
    strides = [1] * len(n_chunks)
    for dim in range(len(n_chunks) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * n_chunks[dim + 1]
    return strides


def _chunk_id(chunk_index, strides):
    return sum(index * stride for index, stride in zip(chunk_index, strides))


def _is_chunk_spec(value):
    if isinstance(value, (int, np.integer)):
        return True
    if not isinstance(value, (tuple, list, np.ndarray)):
        return False
    return all(isinstance(size, (int, np.integer)) for size in value)


def _first_positions(labels, chunk_shape, chunk_start, shape, swaps):
    """Return the first flattened position of each local label.

    The positions are expressed in the same scan order used by
    ``label_cython`` for the full image.  This makes the final labels match
    those produced by a non-chunked call.
    """
    positions = np.full(int(labels.max()) + 1, np.iinfo(np.int64).max, dtype=np.int64)
    if labels.size == 0:
        return positions

    if not swaps:
        values, first_indices = np.unique(labels, return_index=True)
        coordinates = np.unravel_index(first_indices, chunk_shape)
        global_coordinates = [
            coordinate + start
            for coordinate, start in zip(coordinates, chunk_start)
        ]
        positions[values] = np.ravel_multi_index(global_coordinates, shape)
        return positions

    coordinates = list(np.indices(chunk_shape, sparse=True))
    for dim, start in enumerate(chunk_start):
        coordinates[dim] = coordinates[dim] + start

    swapped_shape = list(shape)
    for first_axis, second_axis in swaps:
        coordinates[first_axis], coordinates[second_axis] = (
            coordinates[second_axis],
            coordinates[first_axis],
        )
        swapped_shape[first_axis], swapped_shape[second_axis] = (
            swapped_shape[second_axis],
            swapped_shape[first_axis],
        )

    flattened = np.zeros(chunk_shape, dtype=np.int64)
    for dim, coordinate in enumerate(coordinates):
        flattened = flattened * swapped_shape[dim] + coordinate
    np.minimum.at(positions, labels, flattened)
    return positions


class _UnionFind:
    def __init__(self):
        self.parent = []
        self.size = []

    def add(self, count):
        start = len(self.parent)
        self.parent.extend(range(start, start + count))
        self.size.extend([1] * count)
        return start

    def find(self, node):
        parent = self.parent
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(self, first, second):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]


def _validate_output(label_image, out):
    if tuple(out.shape) != tuple(label_image.shape):
        raise ValueError(
            f"out must have the same shape as label_image, got {out.shape} "
            f"and {label_image.shape}."
        )
    if out.dtype.kind not in "iu":
        raise TypeError("out must have an integer dtype")
    if hasattr(out, "flags") and not out.flags.writeable:
        raise ValueError("out must be writable")
    if out is label_image:
        raise ValueError("out must be distinct from label_image")
    if isinstance(out, np.ndarray) and isinstance(label_image, np.ndarray):
        if np.may_share_memory(out, label_image):
            raise ValueError("out must not overlap label_image")


def _union_boundaries(
    label_image,
    out,
    chunk_index,
    intervals,
    chunk_strides,
    bases,
    current_base,
    current_labels,
    current_input,
    background,
    connectivity,
    union_find,
):
    ndim = len(chunk_index)
    background_value = 0 if background is None else background
    current_id = _chunk_id(chunk_index, chunk_strides)

    for chunk_delta in itertools.product((-1, 0, 1), repeat=ndim):
        n_axes = sum(offset != 0 for offset in chunk_delta)
        if n_axes == 0 or n_axes > connectivity:
            continue

        previous_index = list(chunk_index)
        for dim, offset in enumerate(chunk_delta):
            if offset:
                previous_index[dim] += offset
                if not 0 <= previous_index[dim] < len(intervals[dim]):
                    break
        else:
            previous_id = _chunk_id(previous_index, chunk_strides)
            if previous_id >= current_id:
                continue
            previous_base = bases[previous_id]

            for voxel_delta in itertools.product((-1, 0, 1), repeat=ndim):
                if sum(offset != 0 for offset in voxel_delta) > connectivity:
                    continue
                if any(
                    chunk_offset and voxel_offset != -chunk_offset
                    for chunk_offset, voxel_offset in zip(
                        chunk_delta, voxel_delta
                    )
                ):
                    continue

                previous_slices = []
                current_slices = []
                empty = False
                for dim, (chunk_offset, voxel_offset) in enumerate(
                    zip(chunk_delta, voxel_delta)
                ):
                    start, stop = intervals[dim][chunk_index[dim]]
                    length = stop - start
                    if chunk_offset == -1:
                        previous_slices.append(slice(start - 1, start))
                        current_slices.append(slice(0, 1))
                    elif chunk_offset == 1:
                        previous_slices.append(slice(stop, stop + 1))
                        current_slices.append(slice(length - 1, length))
                    elif voxel_offset == 0:
                        previous_slices.append(slice(start, stop))
                        current_slices.append(slice(0, length))
                    elif voxel_offset > 0:
                        previous_slices.append(slice(start, stop - 1))
                        current_slices.append(slice(1, length))
                        empty |= length <= 1
                    else:
                        previous_slices.append(slice(start + 1, stop))
                        current_slices.append(slice(0, length - 1))
                        empty |= length <= 1

                if empty:
                    continue

                previous_slices = tuple(previous_slices)
                current_slices = tuple(current_slices)
                previous_input = np.asarray(label_image[previous_slices])
                previous_labels = np.asarray(out[previous_slices])
                adjacent_input = current_input[current_slices]
                adjacent_labels = current_labels[current_slices]

                contact = (
                    (previous_labels > 0)
                    & (adjacent_labels > 0)
                    & (previous_input == adjacent_input)
                    & (previous_input != background_value)
                )
                if not np.any(contact):
                    continue

                pairs = np.unique(
                    np.column_stack(
                        (previous_labels[contact], adjacent_labels[contact])
                    ),
                    axis=0,
                )
                for previous_label, adjacent_label in pairs:
                    union_find.union(
                        previous_base + int(previous_label) - 1,
                        current_base + int(adjacent_label) - 1,
                    )


def _label_chunked(
    label_image,
    chunks,
    out,
    background=None,
    return_num=False,
    connectivity=None,
):
    shape = tuple(label_image.shape)
    ndim = len(shape)
    if ndim == 0:
        raise ValueError("labeling a 0-dimensional image is not supported")

    chunks = _normalize_chunks(chunks, shape)
    intervals = _chunk_intervals(shape, chunks)
    n_chunks = tuple(len(axis_intervals) for axis_intervals in intervals)

    if connectivity is None:
        connectivity = ndim
    if not 1 <= connectivity <= ndim:
        raise ValueError(
            f"Connectivity for {ndim}D image should "
            f"be in [1, ..., {ndim}]. Got {connectivity}."
        )

    if out is None:
        out = np.empty(shape, dtype=np.intp)
    _validate_output(label_image, out)

    if any(axis_n_chunks == 0 for axis_n_chunks in n_chunks):
        if return_num:
            return out, 0
        return out

    swaps = [] if label_image.dtype == bool else _get_swaps(shape)

    chunk_strides = _chunk_strides(n_chunks)
    union_find = _UnionFind()
    bases = []
    local_label_counts = []
    max_output_label = np.iinfo(out.dtype).max

    for chunk_index in itertools.product(*(range(count) for count in n_chunks)):
        slices = _chunk_slices(intervals, chunk_index)
        current_input = np.asarray(label_image[slices])
        current_labels, n_labels = label(
            current_input,
            background=background,
            return_num=True,
            connectivity=connectivity,
        )
        if n_labels > max_output_label:
            raise ValueError(
                "out dtype is too small to hold the local labels needed for "
                "chunked labeling."
            )

        current_base = union_find.add(n_labels)
        bases.append(current_base)
        local_label_counts.append(n_labels)

        _union_boundaries(
            label_image,
            out,
            chunk_index,
            intervals,
            chunk_strides,
            bases,
            current_base,
            current_labels,
            current_input,
            background,
            connectivity,
            union_find,
        )
        out[slices] = current_labels

    root_positions = {}
    for chunk_index in itertools.product(*(range(count) for count in n_chunks)):
        slices = _chunk_slices(intervals, chunk_index)
        chunk_id = _chunk_id(chunk_index, chunk_strides)
        current_base = bases[chunk_id]
        n_labels = local_label_counts[chunk_id]
        current_labels = np.asarray(out[slices])

        positions = _first_positions(
            current_labels,
            _chunk_shape(intervals, chunk_index),
            tuple(intervals[dim][chunk_index[dim]][0] for dim in range(ndim)),
            shape,
            swaps,
        )
        for local_label in range(1, n_labels + 1):
            root = union_find.find(current_base + local_label - 1)
            position = int(positions[local_label])
            if root not in root_positions or position < root_positions[root]:
                root_positions[root] = position

    ordered_roots = sorted(root_positions, key=root_positions.__getitem__)
    global_labels = {
        root: global_label for global_label, root in enumerate(ordered_roots, start=1)
    }
    n_final_labels = len(global_labels)
    if n_final_labels > max_output_label:
        raise ValueError("out dtype is too small to hold the labels")

    for chunk_index in itertools.product(*(range(count) for count in n_chunks)):
        slices = _chunk_slices(intervals, chunk_index)
        chunk_id = _chunk_id(chunk_index, chunk_strides)
        current_base = bases[chunk_id]
        n_labels = local_label_counts[chunk_id]
        current_labels = np.asarray(out[slices])

        lookup = np.zeros(n_labels + 1, dtype=np.int64)
        for local_label in range(1, n_labels + 1):
            root = union_find.find(current_base + local_label - 1)
            lookup[local_label] = global_labels[root]

        out[slices] = lookup[current_labels]

    if return_num:
        return out, n_final_labels
    return out


def _label_bool(image, background=None, return_num=False, connectivity=None):
    """Faster implementation of clabel for boolean input.

    See context: https://github.com/scikit-image/scikit-image/issues/4833
    """
    from ..morphology._util import _resolve_neighborhood

    if background == 1:
        image = ~image

    if connectivity is None:
        connectivity = image.ndim

    if not 1 <= connectivity <= image.ndim:
        raise ValueError(
            f'Connectivity for {image.ndim}D image should '
            f'be in [1, ..., {image.ndim}]. Got {connectivity}.'
        )

    footprint = _resolve_neighborhood(None, connectivity, image.ndim)
    result = ndimage.label(image, structure=footprint)

    if return_num:
        return result
    else:
        return result[0]


def label(
    label_image,
    background=None,
    return_num=False,
    connectivity=None,
    *,
    chunks=None,
    out=None,
    chunk_size=None,
):
    r"""Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    In 2D, they can be neighbors either in a 1- or 2-connected sense.
    The value refers to the maximum number of orthogonal hops to consider a
    pixel/voxel a neighbor::

      1-connectivity     2-connectivity     diagonal connection close-up

           [ ]           [ ]  [ ]  [ ]             [ ]
            |               \  |  /                 |  <- hop 2
      [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
            |               /  |  \             hop 1
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    label_image : ndarray of dtype int
        Image to label.
    background : int, optional
        Consider all pixels with this value as background pixels, and label
        them as 0. By default, 0-valued pixels are considered as background
        pixels.
    return_num : bool, optional
        Whether to return the number of assigned labels.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.
    chunks : int or tuple of int, optional
        If provided, label the image blockwise using chunks of this shape.
        The input is only accessed through slices of at most this size, and
        results are written to ``out`` block by block.
    out : ndarray, optional
        Output array used when ``chunks`` is provided. It must have the same
        shape as ``label_image`` and an integer dtype. A memory-mapped array
        is supported.
    chunk_size : int or tuple of int, optional
        Alias for ``chunks``.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.
    num : int, optional
        Number of labels, which equals the maximum label index and is only
        returned if return_num is `True`.

    See Also
    --------
    label_chunked
    skimage.measure.regionprops
    skimage.measure.regionprops_table

    References
    ----------
    .. [1] Christophe Fiorio and Jens Gustedt, "Two linear time Union-Find
           strategies for image processing", Theoretical Computer Science
           154 (1996), pp. 165-181.
    .. [2] Kensheng Wu, Ekow Otoo and Arie Shoshani, "Optimizing connected
           component labeling algorithms", Paper LBNL-56864, 2005,
           Lawrence Berkeley National Laboratory (University of California),
           http://repositories.cdlib.org/lbnl/LBNL-56864

    Examples
    --------
    >>> import numpy as np
    >>> x = np.eye(3).astype(int)
    >>> print(x)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, connectivity=1))
    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    >>> print(label(x, connectivity=2))
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, background=-1))
    [[1 2 2]
     [2 1 2]
     [2 2 1]]
    >>> x = np.array([[1, 0, 0],
    ...               [1, 1, 5],
    ...               [0, 0, 0]])
    >>> print(label(x))
    [[1 0 0]
     [1 1 2]
     [0 0 0]]
    """
    if chunk_size is not None:
        if chunks is not None:
            raise ValueError("Specify only one of chunks and chunk_size.")
        chunks = chunk_size

    if chunks is not None:
        return _label_chunked(
            label_image,
            chunks,
            out,
            background=background,
            return_num=return_num,
            connectivity=connectivity,
        )
    if out is not None:
        raise ValueError("out requires chunks; use label_chunked for blockwise I/O.")

    if label_image.dtype == bool:
        return _label_bool(
            label_image,
            background=background,
            return_num=return_num,
            connectivity=connectivity,
        )
    else:
        return clabel(label_image, background, return_num, connectivity)


def label_chunked(
    label_image,
    *args,
    chunks=None,
    out=None,
    background=None,
    return_num=False,
    connectivity=None,
    chunk_size=None,
):
    """Label an array in chunks, optionally writing to a memory-mapped array.

    This function is equivalent to :func:`label`, but processes the input in
    blocks. Only one input block and its corresponding output block are held
    in memory at a time. Components that touch across block faces, edges, or
    corners are joined before the final labels are written back.

    Parameters
    ----------
    label_image : array-like
        Integer array supporting slicing and exposing ``shape``. It is never
        modified.
    chunks : int or tuple of int
        Size of the chunks along each dimension. An integer is used for every
        dimension.
    out : ndarray, optional
        Output array with the same shape as ``label_image`` and an integer
        dtype. Memory-mapped arrays are supported. If omitted, an in-memory
        array is allocated.
    background : int, optional
        Value treated as background, as in :func:`label`.
    return_num : bool, optional
        Whether to return the number of assigned labels.
    connectivity : int, optional
        Neighbor connectivity, as in :func:`label`.
    chunk_size : int or tuple of int, optional
        Alias for ``chunks``.

    See Also
    --------
    label

    Returns
    -------
    labels : ndarray
        The ``out`` array, filled with labels.
    num : int, optional
        Number of labels, returned if ``return_num`` is ``True``.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.zeros((4, 4, 4), dtype=np.int8)
    >>> image[1:3, 1:3, 1:3] = 7
    >>> out = np.empty(image.shape, dtype=np.int32)
    >>> labels = label_chunked(image, chunks=(2, 2, 2), out=out)
    >>> np.array_equal(labels, label(image))
    True
    """
    if len(args) > 2:
        raise TypeError("label_chunked accepts at most two positional arguments")
    for argument in args:
        if (
            hasattr(argument, "shape")
            and tuple(argument.shape) == tuple(label_image.shape)
            and not _is_chunk_spec(argument)
        ):
            if out is not None:
                raise TypeError("out was specified more than once")
            out = argument
        elif _is_chunk_spec(argument):
            if chunks is not None:
                raise TypeError("chunks was specified more than once")
            chunks = argument
        else:
            raise TypeError("chunks must be an integer or a tuple of integers")

    if chunk_size is not None:
        if chunks is not None:
            raise ValueError("Specify only one of chunks and chunk_size.")
        chunks = chunk_size

    if chunks is None:
        raise ValueError("chunks must be provided for chunked labeling")

    return _label_chunked(
        label_image,
        chunks,
        out,
        background=background,
        return_num=return_num,
        connectivity=connectivity,
    )
