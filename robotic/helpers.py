import contextlib
import functools
import os
from typing import ClassVar

import numpy as np
import trimesh

from robotic._robotic import KOMO, Config


@functools.cache
def getenv(key: str, default=0):
    return type(default)(os.getenv(key, default))


class Context(contextlib.ContextDecorator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.old_context: dict[str, int] = {k: v.value for k, v in ContextVar._cache.items()}
        for k, v in self.kwargs.items():
            ContextVar._cache[k].value = v

    def __exit__(self, *args):
        for k, v in self.old_context.items():
            ContextVar._cache[k].value = v


class ContextVar:
    _cache: ClassVar[dict[str, "ContextVar"]] = {}
    value: int
    key: str

    def __init__(self, key, default_value):
        if key in ContextVar._cache:
            raise RuntimeError(f"attempt to recreate ContextVar {key}")
        ContextVar._cache[key] = self
        self.value, self.key = getenv(key, default_value), key

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, x):
        return self.value >= x

    def __gt__(self, x):
        return self.value > x

    def __lt__(self, x):
        return self.value < x


DEBUG = ContextVar("DEBUG", 0)


def get_mesh(config: Config, frame_name: str, colors=True, transform=True) -> trimesh.Trimesh:
    frame = config.getFrame(frame_name)
    points, triangles, face_colors = frame.getMesh()
    if not colors or face_colors.ndim == 0:
        face_colors = None
    if transform:
        return trimesh.Trimesh(vertices=points, faces=triangles, face_colors=face_colors).apply_transform(frame.getTransform())
    else:
        return trimesh.Trimesh(vertices=points, faces=triangles, face_colors=face_colors)


def create_gripper_mesh(width=0.1, thickness=0.01, finger_length=0.04) -> trimesh.Trimesh:
    base = trimesh.creation.box(extents=[width, thickness, thickness])
    finger1 = trimesh.creation.box(extents=[thickness, thickness, finger_length])
    finger2 = trimesh.creation.box(extents=[thickness, thickness, finger_length])

    finger_offset_y = (width - thickness) / 2
    finger_offset_z = -(thickness + finger_length) / 2
    finger1.apply_translation([finger_offset_y, 0, finger_offset_z])
    finger2.apply_translation([-finger_offset_y, 0, finger_offset_z])

    # the square at the tip of the fingers is the center (thickness x thickness x thickness)
    gripper = trimesh.util.concatenate([base, finger1, finger2])
    gripper.apply_translation([0, 0, finger_length - thickness])
    return gripper


def config_to_trimesh(config: Config):
    meshes = []
    for frame in config.getFrames():
        print(frame.name)
        try:
            meshes.append(get_mesh(config, frame.name))
        except IndexError:
            print("no idea what this index error is")
        except OverflowError:
            print("overflow wtf")
    return meshes


def komo_to_trimesh(komo: KOMO, phase: int):
    meshes = []
    for frameName in komo.getConfig().getFrameNames():
        frame = komo.getFrame(frameName, phase)
        try:
            meshes.append(get_mesh(komo.getConfig(), frameName, transform=False).apply_transform(frame.getTransform()))
        except IndexError:
            print("no idea what this index error is")
        except OverflowError:
            print("overflow wtf")


def compute_look_at_matrix(origin_pos: np.typing.ArrayLike, target_pos: np.typing.ArrayLike):
    """Return a rotation matrix that orients from camera_pos to face target_pos."""
    if abs(origin_pos[0]) < np.finfo(np.float32).eps:  # weird robotic behaviour: avoid x being too close to zero
        origin_pos[0] = np.finfo(np.float32).eps
    forward = target_pos - origin_pos
    forward /= np.linalg.norm(forward)
    right = np.cross([0, 0, -1], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.column_stack((right, up, forward))


def mask_colors(seg_rgb: np.ndarray, colors: np.ndarray):
    """Return mask where True = pixel matches one of the given colors."""
    seg_int = (seg_rgb[:, :, 0].astype(np.uint32) << 16) | (seg_rgb[:, :, 1].astype(np.uint32) << 8) | seg_rgb[:, :, 2].astype(np.uint32)
    colors_int = (colors[:, 0].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 2].astype(np.uint32)
    return np.isin(seg_int, colors_int)


def rgb_to_gray(image: np.ndarray):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
