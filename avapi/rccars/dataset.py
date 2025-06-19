import os
import numpy as np
import json
from cv2 import imread

from avstack.config import DATASETS
from avstack.geometry import GlobalOrigin3D, ReferenceFrame, q_stan_to_cam, Box2D
from avstack.calibration import CameraCalibration
from .._dataset import BaseSceneDataset, BaseSceneManager


_nominal_whitelist_types = ["car"]
_nominal_ignore_types = ["car"]


@DATASETS.register_module()
class RcCarsScenesManager(BaseSceneManager):
    name = "rccars"
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(
        self,
        data_dir: str,
        seed: int = 1,
        verbose: bool = False,
    ):
        if not os.path.exists(data_dir):
            raise RuntimeError(f"Cannot find data dir at {data_dir}")
        self.data_dir = data_dir
        self.verbose = verbose
        self.scenes = sorted(next(os.walk(data_dir))[1])

    def get_scene_dataset_by_index(self, scene_idx, split):
        return RcCarsSceneDataset(self.data_dir, self.scenes[scene_idx], split)

    def get_scene_dataset_by_name(self, scene_name, split):
        if not scene_name in self.scenes:
            raise IndexError(f"Cannot find scene {scene_name} in {self.scenes}")
        return RcCarsSceneDataset(self.data_dir, scene_name, split)



class RcCarsSceneDataset(BaseSceneDataset):
    name = "rccars"
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(
        self,
        data_dir: str,
        scene: str,
        split: str,
        whitelist_types: list = _nominal_whitelist_types,
        ignore_types: list = _nominal_ignore_types,
    ):
        # avstack fields
        self.data_dir = data_dir
        self.scene = scene
        self.split = split
        self.sequence_id = scene
        self.sensors = {"main_camera": "camera"}
        self.sensor_IDs = {"camera": "images"}
        self.scene_path = os.path.join(data_dir, scene, split)
        super().__init__(whitelist_types, ignore_types)

        # load in the json
        with open(os.path.join(self.scene_path, f"{split}.json")) as f:
            self._result = json.load(f)

        # create variables needed for processing
        self._img_id_to_file = {
            img["id"]: img["file_name"]
            for img in self._result["images"]
        }
        self._img_id_to_anns = {
            img["id"]: []
            for img in self._result["images"]
        }
        for ann in self._result["annotations"]:
            self._img_id_to_anns[ann["image_id"]].append(ann)
        self._obj_id_to_str = {
            cat["id"]: cat["name"]
            for cat in self._result["categories"]
        }

    @property
    def frames(self):
        return sorted(list(self._img_id_to_file.keys())) 

    def __str__(self):
        return f"RCCar Dataset of folder: {self.scene_path}"
    
    def _load_timestamp(self, *args, **kwargs):
        return 0.0  # TODO
        
    def _load_calibration(self, frame: int, sensor: str, *args, **kwargs):
        reference = ReferenceFrame(x=np.zeros((3,)), q=q_stan_to_cam, reference=GlobalOrigin3D)
        calib = CameraCalibration(
            reference=reference,
            P=np.zeros((3, 4)),  # TODO
            img_shape=[100, 100],  # TODO
        )
        return calib

    def _load_image(self, frame: int, sensor: str, **kwargs):
        file_name = self._img_id_to_file[frame]
        # img_folder = "images" if sensor is None else self.sensor_IDs[sensor]
        full_path = os.path.join(self.scene_path, file_name)

        if not os.path.exists(full_path):
            raise FileNotFoundError(full_path)
        
        return imread(full_path)
    
    def get_boxes(self, frame: int, sensor: str = None, **kwargs):
        calib = self.get_calibration(frame=frame, sensor=sensor)
        boxes = [
            Box2D(
                box2d=[
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ],
                calibration=calib,
                obj_type=self._obj_id_to_str[ann["category_id"]]
            )
            for ann in self._img_id_to_anns[frame]
        ]
        return boxes
    
    def get_ego_reference(self, *args, **kwargs):
        return GlobalOrigin3D
        