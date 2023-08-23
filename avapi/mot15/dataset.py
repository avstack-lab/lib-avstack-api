import configparser
import glob
import os
import re
import shutil
from datetime import datetime

import numpy as np
from avstack import calibration
from avstack.geometry import GlobalOrigin3D, Box2D, Position
from avstack.environment.objects import ObjectState
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
        self.scenes_with_3d = ["ETH-Bahnhof", "ETH-Sunnyday", "PETS09-S2L1", "TUD-Stadtmitte"]

    def get_scene_dataset_by_name(self, scene):
        return Mot15SceneDataset(self.data_dir, self.split, scene)

    def get_scene_dataset_by_index(self, scene_idx):
        return self.get_scene_dataset_by_name(self.scenes[scene_idx])


class Mot15SceneDataset(BaseSceneDataset):
    NAME = "MOT15"
    sensors = {"main_camera": "img1", "camera": "img1"}
    sensor_IDs = {"img1": 1}

    def __init__(
        self,
        data_dir,
        split,
        scene,
        whitelist_types=_nominal_whitelist_types,
        ignore_types=_nominal_ignore_types,
    ):
        super().__init__(whitelist_types, ignore_types)
        self.data_dir = data_dir
        self.split = split
        self.scene = scene
        self.postfix = "img1"
        self.split_path = os.path.join(
            self.data_dir, self.split, self.scene, self.postfix
        )
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
        self.frames = [
            int(img.replace(self.split_path, "").strip("/").split(".")[0])
            for img in self.images
        ]
        self._frame_to_idx_map = {frame: i for i, frame in enumerate(self.frames)}

        # calibration will always be fixed for each scene
        origin = GlobalOrigin3D
        img_shape = (
            self.seqinfo.getint("Sequence", "imHeight"),
            self.seqinfo.getint("Sequence", "imWidth"),
            3,
        )
        fx = 1440  # TODO
        fy = 1440  # TODO
        u = img_shape[1] / 2
        v = img_shape[0] / 2
        P = np.array([[fx, 0, u, 0], [0, fy, v, 0], [0, 0, 1, 0]])
        self.calibration = calibration.CameraCalibration(
            origin, P, img_shape, channel_order="bgr"
        )
        self.framerate = self.seqinfo.getfloat("Sequence", "frameRate")
        self.interval = 1.0 / self.framerate

        # load in ground truth
        self.has_3d = False
        if self.split != "test":
            gt_path = os.path.join(
                self.data_dir, self.split, self.scene, 'gt', 'gt.txt'
            )
            with open(gt_path, 'r') as f:
                gt_lines = [line.strip() for line in f.readlines()]
            self.gt_dict = {frame: [] for frame in self.frames}
            for line in gt_lines:
                data = [float(item) for item in line.split(',')]
                left, top, width, height = data[2:6]
                conf = data[7]
                box2d = Box2D([left, top, left+width, top+height], calibration=self.calibration, ID=data[1])
                if np.any(np.array(data[8:11]) != -1):
                    pos_3d = Position(data[8:11], reference=origin)
                    self.has_3d = True
                else:
                    pos_3d = None
                    self.has_3d = False
                obj_state = ObjectState(obj_type="pedestrian", ID=data[1])
                obj_state.set(
                    t=self.get_timestamp(data[0]),
                    box=box2d,
                    position=pos_3d,
                )
                self.gt_dict[data[0]].append(obj_state)

    def get_ego_reference(self, *args, **kwargs):
        return GlobalOrigin3D

    def _load_frames(self, **kwargs):
        return self.frames

    def _load_timestamp(self, frame, **kwargs):
        return (frame - 1) * self.interval

    def _load_calibration(self, *args, **kwargs):
        return self.calibration

    def _load_image(self, frame, **kwargs):
        img = imread(self.images[self._frame_to_idx_map[frame]])
        return img

    def _load_objects(self, frame, **kwargs):
        return self.gt_dict[frame]