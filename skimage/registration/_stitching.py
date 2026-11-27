"""Panorama stitching re-exported in the registration namespace."""

from ..transform._stitching import (
    panorama,
    panorama_stitch,
    stitch,
    stitch_images,
)

__all__ = ['stitch_images', 'stitch', 'panorama', 'panorama_stitch']
