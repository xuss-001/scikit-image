:orphan:

scikit-image 0.26.0 (unreleased)
================================

New Features
------------
- A new ``skimage.transform.seam_carve`` function performs content-aware
  image reduction by repeatedly removing low-energy seams. An optional
  boolean protection mask marks content that must never be removed; when
  the requested target size conflicts with the mask, a ``ValueError`` is
  raised instead of degrading the protected region. Only grayscale and RGB
  images and size reductions are supported.

Improvements
------------

API Changes
-----------

Bugfixes
--------

Deprecations
------------

Contributors to this release
----------------------------
