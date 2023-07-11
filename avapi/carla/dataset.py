# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-09-28
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-09-29
# @Description:
"""
CARLA dataset manager on AVstack conventions
"""
import glob
import os

import numpy as np
from avstack import calibration
from avstack.geometry import ReferenceFrame, Rotation, Vector, q_mult_vec
from cv2 import imread, imwrite

from .._dataset import BaseSceneDataset, BaseSceneManager


def check_xor_for_none(a, b):
    if (a is None) and (b is None):
        raise ValueError("Both inputs cannot be none")
    if (a is not None) and (b is not None):
        raise ValueError("At least one input must be none")


def get_splits_scenes(data_dir, modval=4, seed=1):
    CSM = CarlaScenesManager(data_dir)
    return CSM.make_splits_scenes(modval=modval, seed=seed)


# _nominal_whitelist_types = ['car', 'pedestrian', 'bicycle',
#         'truck', 'motorcycle']
_nominal_whitelist_types = ["car", "bicycle", "truck", "motorcycle"]
_nominal_ignore_types = []


class CarlaScenesManager(BaseSceneManager):
    name = "CARLA"
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(self, data_dir, split=None, verbose=False):
        """
        data_dir: the base folder where all scenes are kept
        """
        if not os.path.exists(data_dir):
            raise RuntimeError(f"Cannot find data dir at {data_dir}")
        self.data_dir = data_dir
        self.scenes = sorted(next(os.walk(data_dir))[1])
        self.splits_scenes = self.make_splits_scenes(modval=4, seed=1)

    def get_scene_dataset_by_index(self, scene_idx):
        return CarlaSceneDataset(self.data_dir, self.scenes[scene_idx])

    def get_scene_dataset_by_name(self, scene_name):
        if not scene_name in self.scenes:
            raise IndexError(f"Cannot find scene {scene_name} in {self.scenes}")
        return CarlaSceneDataset(self.data_dir, scene_name)


class CarlaSceneDataset(BaseSceneDataset):
    name = "CARLA"
    CFG = {}
    CFG["num_lidar_features"] = 4
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types
    sensors = {
        "main_lidar": "LIDAR_TOP",
        "lidar": "LIDAR_TOP",
        "main_camera": "CAM_FRONT",
    }

    def __init__(
        self,
        data_dir,
        scene,
        whitelist_types=_nominal_whitelist_types,
        ignore_types=_nominal_ignore_types,
    ):
        self.data_dir = data_dir
        self.scene = scene
        self.sequence_id = scene
        self.scene_path = os.path.join(data_dir, scene)

        # -- object and ego files
        self.obj_folder = os.path.join(self.scene_path, "objects", "avstack")
        self.obj_local_folder = os.path.join(self.scene_path, "objects_sensor")
        ego_files = {"timestamp": {}, "frame": {}}
        npc_files = {"timestamp": {}, "frame": {}}
        ego_frame_to_ts = {}
        npc_frame_to_ts = {}
        for filename in sorted(glob.glob(os.path.join(self.obj_folder, "*.txt"))):
            filename = filename.split("/")[-1]
            ts, frame, _ = filename.split("-")
            ts = float(ts.split("_")[1])
            frame = int(frame.split(".")[0].split("_")[1])
            if "ego" in filename:
                ego_files["timestamp"][ts] = filename
                ego_files["frame"][frame] = filename
                ego_frame_to_ts[frame] = ts
            elif "npc" in filename:
                # -- global frame
                npc_files["timestamp"][ts] = filename
                npc_files["frame"][frame] = filename
                npc_frame_to_ts[frame] = ts
            else:
                raise NotImplementedError(filename)
        self.ego_frame_to_ts = ego_frame_to_ts
        self.npc_frame_to_ts = npc_frame_to_ts
        self.ego_files = ego_files
        self.npc_files = npc_files

        # -- dynamically create sensor ID mappings
        sensor_IDs = {}
        sensor_folders = {}
        sensor_file_post = {}
        sensor_frame_to_ts = {}
        sensor_frames = {}
        file_endings = {}
        sensor_data_folder = os.path.join(self.scene_path, "sensor_data")
        if os.path.exists(sensor_data_folder):
            self.sensor_data_folder = sensor_data_folder
            for sens in sorted(next(os.walk(self.sensor_data_folder))[1]):
                if not len(sens.split("-")) == 2:
                    raise ValueError(f"Cannot understand sensor data folder {sens}")
                name, ID = sens.split("-")
                sensor_IDs[name] = int(ID)
                sensor_folders[name] = os.path.join(sensor_data_folder, sens)
                sensor_file_post[name] = {"timestamp": {}, "frame": {}}
                sensor_frame_to_ts[name] = {}
                sensor_frames[name] = []
                # within each folder, parse the sensor timestamps and frames
                for i, filename in enumerate(
                    sorted(glob.glob(os.path.join(sensor_folders[name], "data-*")))
                ):
                    filename = filename.split("/")[-1]
                    if i == 0:
                        file_endings[name] = "." + filename.split(".")[-1]
                    _, ts, frame = filename.split("-")
                    ts = float(ts.split("_")[1])
                    frame = int(frame.split(".")[0].split("_")[1])
                    fname_post = filename.replace("data-", "")[:-4]
                    if fname_post.endswith("."):
                        fname_post = fname_post[:-1]
                    sensor_file_post[name]["timestamp"][ts] = fname_post
                    sensor_file_post[name]["frame"][frame] = fname_post
                    sensor_frame_to_ts[name][frame] = float(ts)
                    sensor_frames[name].append(frame)
                sensor_frames[name] = sorted(sensor_frames[name])
        self.sensor_folders = sensor_folders
        self.sensor_file_post = sensor_file_post
        self.sensor_frame_to_ts = sensor_frame_to_ts
        self.sensor_frames = sensor_frames
        self.file_endings = file_endings
        self.sensor_IDs = sensor_IDs
        self.framerate = 1 / np.median(
            np.diff(list(self.sensor_frame_to_ts[self.sensors["main_camera"]].values()))
        )
        super().__init__(whitelist_types, ignore_types)

    @property
    def frames(self):
        return list(self.sensor_frames[self.sensors["main_camera"]])

    @property
    def frames_ego(self):
        return list(self.ego_frame_to_ts.keys())

    def __str__(self):
        return f"CARLA Object Datset of folder: {self.scene_path}"

    def get_object_file(self, frame, timestamp, is_ego, is_global, sensor=None):
        check_xor_for_none(frame, timestamp)
        if frame is not None:
            if is_ego:
                file_post = self.ego_files["frame"][frame]
            else:
                file_post = self.npc_files["frame"][frame]
        else:
            raise
        if is_global:
            filepath = os.path.join(self.obj_folder, file_post)
        else:
            assert sensor is not None
            filepath = os.path.join(self.obj_local_folder, sensor, file_post)
        return filepath

    def get_sensor_file(self, frame, timestamp, sensor, file_type):
        check_xor_for_none(frame, timestamp)
        if frame is not None:
            file_post = self.sensor_file_post[sensor]["frame"][frame]
        else:
            # TODO: ALLOW FOR INTERPOLATION OR NEAREST????
            file_post = self.sensor_file_post[sensor]["timestamp"][timestamp]
        filepath = os.path.join(
            self.sensor_folders[sensor], file_type + "-" + file_post
        )
        return filepath

    def _load_sensor_data_filepath(self, frame, sensor):
        return (
            self.get_sensor_file(frame, None, sensor, "data")
            + self.file_endings[sensor]
        )

    def _load_frames(self, sensor: str):
        return self.sensor_frames[sensor]

    def _load_timestamp(self, frame, sensor, utime=False):
        return self.sensor_frame_to_ts[sensor][frame]

    def _load_calibration(self, frame, sensor):
        """
        NOTE: for now calibration is ego-relative...will be fixed eventually

        Therefore, for infra sensors, have to hack the reference since they
        start in global coordinates
        """
        timestamp = None
        filepath = self.get_sensor_file(frame, timestamp, sensor, "calib")
        with open(filepath + ".txt", "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        calib = calibration.read_calibration_from_line(lines[0])

        # -- extra hack if infrastructure sensor -- calib is global_2_sensor
        if "infrastructure" in sensor.lower():
            """
            infrastructure sensor must be put into ego's frame as follows

            NOTE: this assumes that ego.origin is nominal origin standard

            ego.position.vector := x_OR1_2_ego_in_OR1
            ego.attitude.q      := q_OR1_2_ego
            calib.origin.x      := x_OR1_2_sens_in_OR1
            calib.origin.q      := q_OR1_2_sens

            Thus, to get to the ego-relative frame is:
            -- rotation
            q_ego_2_OR1 = q_OR1_2_ego.conjugate()
            q_ego_2_sens = q_OR1_2_sens * q_ego_2_OR1

            -- translation
            x_OR1_2_sens_in_ego = q_mult_vec(q_OR1_2_ego, x_OR1_2_sens_in_OR1)
            x_ego_2_OR1_in_OR1 = -x_OR1_2_ego_in_OR1
            x_ego_2_OR1_in_ego = q_mult_vec(q_OR1_2_ego, x_ego_2_OR1_in_OR1)
            x_ego_2_sens_in_ego = x_OR1_2_sens_in_ego + x_ego_2_OR1_in_ego
            """
            raise NotImplementedError("Have not updated this for refchoc yet")
            # -- get items
            ego = self.get_ego(frame)
            x_OR1_2_ego_in_OR1 = ego.position.vector
            q_OR1_2_ego = ego.attitude.q
            x_OR1_2_sens_in_OR1 = calib.reference.x
            q_OR1_2_sens = calib.reference.q

            # -- rotation
            q_ego_2_OR1 = q_OR1_2_ego.conjugate()
            q_ego_2_sens = q_OR1_2_sens * q_ego_2_OR1
            # -- translation
            x_OR1_2_sens_in_ego = q_mult_vec(q_OR1_2_ego, x_OR1_2_sens_in_OR1)
            x_ego_2_OR1_in_OR1 = -x_OR1_2_ego_in_OR1
            x_ego_2_OR1_in_ego = q_mult_vec(q_OR1_2_ego, x_ego_2_OR1_in_OR1)
            x_ego_2_sens_in_ego = x_OR1_2_sens_in_ego + x_ego_2_OR1_in_ego

            # -- new origin
            pos = Vector(x_ego_2_sens_in_ego)
            calib.origin = ReferenceFrame(x_ego_2_sens_in_ego, q_ego_2_sens)
        return calib

    def _load_image(self, frame, sensor):
        timestamp = None
        filepath = (
            self.get_sensor_file(frame, timestamp, sensor, "data")
            + self.file_endings[sensor]
        )
        assert os.path.exists(filepath), filepath
        try:
            return imread(filepath)
        except TypeError as e:
            print(filepath)
            raise e

    def _load_depth_image(self, frame, sensor):
        timestamp = None
        filepath = (
            self.get_sensor_file(frame, timestamp, sensor, "data")
            + self.file_endings[sensor]
        )
        assert os.path.exists(filepath), filepath
        try:
            return imread(filepath)
        except TypeError as e:
            print(filepath)
            raise e

    def _load_lidar(self, frame, sensor, filter_front, with_panoptic=False):
        timestamp = None
        filepath = (
            self.get_sensor_file(frame, timestamp, sensor, "data")
            + self.file_endings[sensor]
        )
        assert os.path.exists(filepath), filepath
        if filepath.endswith(".ply"):
            pcd = o3d.io.read_point_cloud(filepath)
            pcd = np.asarray(pcd.points)
        else:
            pcd = np.fromfile(filepath, dtype=np.float32).reshape(
                (-1, self.CFG["num_lidar_features"])
            )
        if filter_front:
            return pcd[pcd[:, 0] > 0, :]  # assumes z is forward....
        else:
            return pcd

    def _load_ego(self, frame):
        timestamp = None
        filepath = self.get_object_file(frame, timestamp, is_ego=True, is_global=True)
        with open(filepath, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        return self.parse_label_line(lines[0])

    def _load_objects(
        self,
        frame,
        sensor,
        whitelist_types=["car", "truck", "bicycle", "motorcycle"],
        ignore_types=[],
    ):
        if sensor is None:
            sensor = "CAM_FRONT"
        timestamp = None
        filepath = self.get_object_file(
            frame, timestamp, sensor=sensor, is_ego=False, is_global=False
        )
        objs = self._read_objects(filepath)
        return np.array(
            [
                obj
                for obj in objs
                if ((obj.obj_type in whitelist_types) or (whitelist_types == "all"))
                and (obj.obj_type not in ignore_types)
            ]
        )

    def _load_objects_global(self, frame, whitelist_types="all", ignore_types=[]):
        timestamp = None
        filepath = self.get_object_file(frame, timestamp, is_ego=False, is_global=True)
        objs = self._read_objects(filepath)
        return np.array(
            [
                obj
                for obj in objs
                if ((obj.obj_type in whitelist_types) or (whitelist_types == "all"))
                and (obj.obj_type not in ignore_types)
            ]
        )

    def _read_objects(self, filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()
        objs = []
        for line in lines:
            line = line.rstrip()
            objs.append(self.parse_label_line(line))
        return np.asarray(objs)

    def _save_objects(self, frame, objects, folder, file):
        data_strs = "\n".join([obj.format_as("avstack") for obj in objects])
        with open(os.path.join(folder, file.format("txt")), "w") as f:
            f.write(data_strs)
