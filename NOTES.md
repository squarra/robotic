Some notes for development

## Open3d and python 3.13

open3d doesn't habe python3.13 wheels in their pypi package yet. You can download the wheel files from the GitHub Actions though.

see: https://github.com/isl-org/Open3D/issues/7284

Then unzip and just `pip install open3d-0.19.0-cp313-cp313-manylinux_2_35_x86_64.whl/open3d_cpu-0.19.0-cp313-cp313-manylinux_2_35_x86_64.whl`

## rai patches

The patches I applied to rai with their respective git tags

### v0.1.3

- When deleting a frame with `delFrame()`, the viewer gets the new state as well. This should actually be patched in cpp I think.
- Added the verbose parameter to `getCollisions()`
- Added call to `self->coll_fclReset()` in `getCollisions()`. The FclInterface only gets initialized once for the Config. Frames that get added afterwards, don't et accounted for in the collisions
- Added `ensure_X()` mapping, so I can call it before `computeCollisions()`. When updating the position of a frame with e.g. `setRelativePosition()`, the absolute positions would not be updated automatically before the collisions get calculated.

### v0.1.2

- Added `self.updateConfiguration()` call to `computeSegmentationImage()` and `computeSegmentationID()` mappings. Prior to this I would have to call `computeImageAndDepth()` so that `self.updateConfiguration()` gets called implicitly.

### v0.1.1

- Fixed pybind mapping for `computeSegmentationID()`