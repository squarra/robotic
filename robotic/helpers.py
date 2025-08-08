import contextlib
import functools
import os
from typing import ClassVar

import trimesh

import robotic as ry


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


def get_mesh(config: ry.Config, frame_name: str, colors=True, transform=True) -> trimesh.Trimesh:
    frame = config.getFrame(frame_name)
    points, triangles, face_colors = frame.getMesh()
    if not colors or face_colors.ndim == 0:
        face_colors = None
    if transform:
        return trimesh.Trimesh(vertices=points, faces=triangles, face_colors=face_colors).apply_transform(frame.getTransform())
    else:
        return trimesh.Trimesh(vertices=points, faces=triangles, face_colors=face_colors)


def load_simple_config(obj_pos=[0.0, 0.25, 0.08]):
    config = ry.Config()
    config.addFile(ry.raiPath("scenarios/pandaSingle.g"))
    config.delFrame("panda_collCameraWrist")
    config.addFrame("obj", "table").setJoint(ry.JT.rigid).setShape(ry.ST.ssBox, [0.15, 0.06, 0.06, 0.005]).setRelativePosition(obj_pos).setContact(1)
    return config


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


def config_to_trimesh(config: ry.Config):
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


def komo_to_trimesh(komo: ry.KOMO, phase: int):
    meshes = []
    for frameName in komo.getConfig().getFrameNames():
        frame = komo.getFrame(frameName, phase)
        try:
            meshes.append(get_mesh(komo.getConfig(), frameName, transform=False).apply_transform(frame.getTransform()))
        except IndexError:
            print("no idea what this index error is")
        except OverflowError:
            print("overflow wtf")
