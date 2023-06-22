import glob
import os
import re
import shutil
from datetime import datetime
import configparser

import numpy as np
from avstack import calibration
from avstack.environment.objects import VehicleState
from avstack.geometry import (
    GlobalOrigin3D,
    Rotation,
    Vector,
    ReferenceFrame,
    bbox,
    q_mult_vec,
    q_stan_to_cam,
    transformations as tforms
)
from avstack.utils import check_xor_for_none
from cv2 import imread
from tqdm import tqdm

from .._dataset import BaseSceneDataset, BaseSceneManager


_nominal_whitelist_types = ["Pedestrian"]
_nominal_ignore_types = []
img_exts = [".jpg", ".jpeg", ".png", ".tiff"]


class MOT15SceneManager(BaseSceneManager):
    NAME = "MOT15"

    def __init__(self, data_dir, split="train", verbose=False) -> None:
        self.data_dir = data_dir
        self.split = split
        self.scenes = glob.glob(os.path.join(self.data_dir, self.split, "*"))
        self.scenes = sorted(
            [
                scene.replace(os.path.join(self.data_dir, self.split), "").lstrip("/")
                for scene in self.scenes
            ]
        )
        self.splits_scenes = None  # use the different splits instead
    
    def get_scene_dataset_by_name(self, scene):
        return Mot15SceneDataset(self.data_dir, self.split, scene)

    def get_scene_dataset_by_index(self, scene_idx):
        return self.get_scene_dataset_by_name(self.scenes[scene_idx])


class Mot15SceneDataset(BaseSceneDataset):
    NAME = "MOT15"
    sensors = {"main_camera":"img1", "camera":"img1"}
    sensor_IDs = {"img1":1}
    
    def __init__(self, data_dir, split, scene,
                 whitelist_types=_nominal_whitelist_types,
                 ignore_types=_nominal_ignore_types):
        super().__init__(whitelist_types, ignore_types)
        self.data_dir = data_dir
        self.split = split
        self.scene = scene
        self.postfix = "img1"
        self.split_path = os.path.join(self.data_dir, self.split, self.scene, self.postfix)
        self.seqinfo = configparser.ConfigParser()
        seq_path = os.path.join(self.data_dir, self.split, self.scene, "seqinfo.ini")
        if not os.path.exists(seq_path):
            raise FileNotFoundError(f"Cannot find sequence file at {seq_path}")
        self.seqinfo.read(seq_path)
        self.images = sorted(
            [
                img
                for ext in img_exts
                for img in glob.glob(os.path.join(self.split_path, "*" + ext))
            ]
        )
        self.frames = [int(img.replace(self.split_path, '').strip("/").split(".")[0]) for img in self.images]
        self._frame_to_idx_map = {frame:i for i, frame in enumerate(self.frames)}
        # calibration will always be fixed for each scene
        origin = GlobalOrigin3D
        img_shape = (self.seqinfo.getint("Sequence", "imHeight"),
                     self.seqinfo.getint("Sequence", "imWidth"),
                     3)
        fx = 1440  # TODO
        fy = 1440  # TODO
        u = img_shape[1] / 2
        v = img_shape[0] / 2
        P = np.array([[fx,  0, u, 0],
                      [ 0, fy, v, 0],
                      [ 0,  0, 1, 0]])
        self.calibration = calibration.CameraCalibration(origin, P, img_shape, channel_order="bgr")
        self.framerate = self.seqinfo.getfloat("Sequence", "frameRate")
        self.interval = 1.0/self.framerate

    def _load_frames(self, **kwargs):
        return self.frames
    
    def _load_timestamp(self, frame, **kwargs):
        return (frame-1) * self.interval
    
    def _load_calibration(self, frame, **kwargs):
        return self.calibration
    
    def _load_image(self, frame, **kwargs):
        img = imread(self.images[self._frame_to_idx_map[frame]])
        return img

    def _load_objects(self, frame, **kwargs):
        raise NotImplementedError("Have not allowed for loading objects yet")