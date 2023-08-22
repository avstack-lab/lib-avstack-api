# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-09-05
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-29
# @Description:
"""

"""
import logging
import os
import struct
from typing import List

import numpy as np
from avstack.geometry.transformations import matrix_cartesian_to_spherical
from scipy.interpolate import interp1d

from .._dataset import (
    _nominal_ignore_types,
    _nominal_whitelist_types,
    _nuBaseDataset,
    _nuManager,
)


try:
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes as create_splits_scenes_nusc

    splits_scenes = create_splits_scenes_nusc()
except ModuleNotFoundError as e:
    print("Cannot import nuscenes")
    splits_scenes = None


class nuScenesManager(_nuManager):
    NAME = "nuScenes"

    def __init__(self, data_dir, split="v1.0-mini", verbose=False):
        nusc = NuScenes(version=split, dataroot=data_dir, verbose=verbose)
        try:
            nusc_can = NuScenesCanBus(dataroot=data_dir)
        except Exception as e:
            logging.warning("Cannot find CAN bus data")
            nusc_can = None
        super().__init__(nusc, nusc_can, data_dir, split, verbose)
        self.scene_name_to_index = {}
        self.scene_number_to_index = {}
        self.index_to_scene = {}
        self.scenes = [sc['name'] for sc in nusc.scene]
        for i, sc in enumerate(nusc.scene):
            self.scene_name_to_index[sc["name"]] = i
            self.scene_number_to_index[int(sc["name"].replace("scene-", ""))] = i
            self.index_to_scene[i] = sc["name"]

    def list_scenes(self):
        self.nuX.list_scenes()

    def get_splits_scenes(self):
        return {
            k: [int(v.split("-")[1]) for v in vs] for k, vs in splits_scenes.items()
        }

    def get_scene_dataset_by_name(self, scene_name):
        idx = self.scene_name_to_index[scene_name]
        return self.get_scene_dataset_by_index(idx)

    def get_scene_dataset_by_scene_number(self, scene_number):
        idx = self.scene_number_to_index[scene_number]
        return self.get_scene_dataset_by_index(idx)

    def get_scene_dataset_by_index(self, scene_idx):
        return nuScenesSceneDataset(
            self.data_dir,
            self.split,
            self.nuX.scene[scene_idx],
            nusc=self.nuX,
            nusc_can=self.nuX_can,
        )


class nuScenesSceneDataset(_nuBaseDataset):
    NAME = "nuScenes"
    CFG = {}
    CFG["num_lidar_features"] = 5
    CFG["IMAGE_WIDTH"] = 1600
    CFG["IMAGE_HEIGHT"] = 900
    CFG["IMAGE_CHANNEL"] = 3
    img_shape = (CFG["IMAGE_HEIGHT"], CFG["IMAGE_WIDTH"], CFG["IMAGE_CHANNEL"])
    sensors = {
        "lidar": "LIDAR_TOP",
        "main_lidar": "LIDAR_TOP",
        "main-lidar": "LIDAR_TOP",
        "main_camera": "CAM_FRONT",
        "main-camera": "CAM_FRONT",
        "radar": "RADAR_FRONT",
        "main_radar": "RADAR_FRONT",
        "main-radar": "RADAR_FRONT",
    }
    keyframerate = 2
    sensor_IDs = {
        "CAM_BACK": 3,
        "CAM_BACK_LEFT": 1,
        "CAM_BACK_RIGHT": 2,
        "CAM_FRONT": 0,
        "CAM_FRONT_LEFT": 4,
        "CAM_FRONT_RIGHT": 5,
        "LIDAR_TOP": 0,
        "RADAR_BACK_LEFT": 0,
        "RADAR_BACK_RIGHT": 1,
        "RADAR_FRONT": 2,
        "RADAR_FRONT_LEFT": 3,
        "RADAR_FRONT_RIGHT": 4,
    }

    def __init__(
        self,
        data_dir,
        split,
        scene,
        nusc=None,
        nusc_can=None,
        verbose=False,
        whitelist_types=_nominal_whitelist_types,
        ignore_types=_nominal_ignore_types,
    ):
        nusc = (
            nusc
            if nusc is not None
            else NuScenes(version=split, dataroot=data_dir, verbose=verbose)
        )
        try:
            nusc_can = (
                nusc_can if nusc_can is not None else NuScenesCanBus(dataroot=data_dir)
            )
        except Exception as e:
            logging.warning("Cannot find CAN bus data")
            nusc_can = None
        self.scene = scene
        self.scene_name = self.scene
        self.framerate = 2
        self.sequence_id = scene["name"]
        self.splits_scenes = splits_scenes
        try:
            veh_speed = self.nuX_can.get_messages(self.scene["name"], "vehicle_monitor")
        except Exception as e:
            self.ego_speed_interp = None
        else:
            veh_speed = np.array([(m["utime"], m["vehicle_speed"]) for m in veh_speed])
            veh_speed[:, 1] *= 1 / 3.6
            self.ego_speed_interp = interp1d(
                veh_speed[:, 0] / 1e6 - self.t0,
                veh_speed[:, 1],
                fill_value="extrapolate",
            )
        super().__init__(
            nusc, nusc_can, data_dir, split, verbose, whitelist_types, ignore_types
        )

    def make_sample_records(self):
        self.sample_records = {
            0: self.nuX.get("sample", self.scene["first_sample_token"])
        }
        for i in range(1, self.scene["nbr_samples"], 1):
            self.sample_records[i] = self.nuX.get(
                "sample", self.sample_records[i - 1]["next"]
            )
        self.t0 = self.sample_records[0]["timestamp"] / 1e6

    def _load_lidar(
        self, frame, sensor="LIDAR_TOP", filter_front=False, with_panoptic=False
    ):
        if sensor.lower() == "lidar":
            sensor = self.sensors["lidar"]
        lidar_fname = self._get_sensor_file_name(frame, sensor)
        lidar = np.fromfile(lidar_fname, dtype=np.float32).reshape(
            (-1, self.CFG["num_lidar_features"])
        )
        if with_panoptic:
            lidar = np.concatenate(
                (lidar, self._load_panoptic_lidar_labels(frame, sensor)[:, None]),
                axis=1,
            )
        if filter_front:
            return lidar[lidar[:, 1] > 0, :]  # y is front on nuscenes
        else:
            return lidar

    def _load_radar(self, frame, sensor="RADAR_FRONT"):
        """ "
        Important: when trying to incorporate the velocity component of
        the radar data, make sure to look at the nuScenes forums for
        an understanding of how to get radial velocity
        """
        radar_fname = self._get_sensor_file_name(frame, sensor)
        pts = load_radar(radar_fname, dynprop_states=[0, 2, 6])
        razel = matrix_cartesian_to_spherical(pts[[0, 1, 2], :].T)
        rrt = np.linalg.norm(pts[[6, 7], :], axis=0)
        return np.concatenate((razel, rrt[:, None]), axis=1)  # Nx4

    def _load_panoptic_lidar_labels(self, frame, sensor="LIDAR_TOP"):
        record = self.nuX.get(
            "panoptic", self._get_sensor_record(frame, sensor)["token"]
        )
        fname = os.path.join(self.data_dir, record["filename"])
        panoptic_labels = (np.load(fname)["data"] // 1000).astype(int)
        return panoptic_labels


def load_radar(
    file_name: str,
    invalid_states: List[int] = [0],
    dynprop_states: List[int] = range(7),
    ambig_states: List[int] = [3],
):
    """
    Loads RADAR data from a Point Cloud Data file. See details below.
    :param file_name: The path of the pointcloud file.
    :param invalid_states: Radar states to be kept. See details below.
    :param dynprop_states: Radar states to be kept. Use [0, 2, 6] for moving objects only. See details below.
    :param ambig_states: Radar states to be kept. See details below.
    To keep all radar returns, set each state filter to range(18).
    :return: <np.float: d, n>. Point cloud matrix with d dimensions and n points.
    Example of the header fields:
    # .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
    SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
    TYPE F F F I I F F F F F I I I I I I I I
    COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    WIDTH 125
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS 125
    DATA binary
    Below some of the fields are explained in more detail:
    x is front, y is left
    vx, vy are the velocities in m/s.
    vx_comp, vy_comp are the velocities in m/s compensated by the ego motion.
    We recommend using the compensated velocities.
    invalid_state: state of Cluster validity state.
    (Invalid states)
    0x01	invalid due to low RCS
    0x02	invalid due to near-field artefact
    0x03	invalid far range cluster because not confirmed in near range
    0x05	reserved
    0x06	invalid cluster due to high mirror probability
    0x07	Invalid cluster because outside sensor field of view
    0x0d	reserved
    0x0e	invalid cluster because it is a harmonics
    (Valid states)
    0x00	valid
    0x04	valid cluster with low RCS
    0x08	valid cluster with azimuth correction due to elevation
    0x09	valid cluster with high child probability
    0x0a	valid cluster with high probability of being a 50 deg artefact
    0x0b	valid cluster but no local maximum
    0x0c	valid cluster with high artefact probability
    0x0f	valid cluster with above 95m in near range
    0x10	valid cluster with high multi-target probability
    0x11	valid cluster with suspicious angle
    dynProp: Dynamic property of cluster to indicate if is moving or not.
    0: moving
    1: stationary
    2: oncoming
    3: stationary candidate
    4: unknown
    5: crossing stationary
    6: crossing moving
    7: stopped
    ambig_state: State of Doppler (radial velocity) ambiguity solution.
    0: invalid
    1: ambiguous
    2: staggered ramp
    3: unambiguous
    4: stationary candidates
    pdh0: False alarm probability of cluster (i.e. probability of being an artefact caused by multipath or similar).
    0: invalid
    1: <25%
    2: 50%
    3: 75%
    4: 90%
    5: 99%
    6: 99.9%
    7: <=100%
    """

    assert file_name.endswith(".pcd"), "Unsupported filetype {}".format(file_name)

    meta = []
    with open(file_name, "rb") as f:
        for line in f:
            line = line.strip().decode("utf-8")
            meta.append(line)
            if line.startswith("DATA"):
                break

        data_binary = f.read()

    # Get the header rows and check if they appear as expected.
    assert meta[0].startswith("#"), "First line must be comment"
    assert meta[1].startswith("VERSION"), "Second line must be VERSION"
    sizes = meta[3].split(" ")[1:]
    types = meta[4].split(" ")[1:]
    counts = meta[5].split(" ")[1:]
    width = int(meta[6].split(" ")[1])
    height = int(meta[7].split(" ")[1])
    data = meta[10].split(" ")[1]
    feature_count = len(types)
    assert width > 0
    assert len([c for c in counts if c != c]) == 0, "Error: COUNT not supported!"
    assert height == 1, "Error: height != 0 not supported!"
    assert data == "binary"

    # Lookup table for how to decode the binaries.
    unpacking_lut = {
        "F": {2: "e", 4: "f", 8: "d"},
        "I": {1: "b", 2: "h", 4: "i", 8: "q"},
        "U": {1: "B", 2: "H", 4: "I", 8: "Q"},
    }
    types_str = "".join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

    # Decode each point.
    offset = 0
    point_count = width
    points = []
    for i in range(point_count):
        point = []
        for p in range(feature_count):
            start_p = offset
            end_p = start_p + int(sizes[p])
            assert end_p < len(data_binary)
            point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
            point.append(point_p)
            offset = end_p
        points.append(point)

    # A NaN in the first point indicates an empty pointcloud.
    point = np.array(points[0])
    if np.any(np.isnan(point)):
        return cls(np.zeros((feature_count, 0)))

    # Convert to numpy matrix.
    points = np.array(points).transpose()

    # If no parameters are provided, use default settings.
    # invalid_states = cls.invalid_states if invalid_states is None else invalid_states
    # dynprop_states = cls.dynprop_states if dynprop_states is None else dynprop_states
    # ambig_states = cls.ambig_states if ambig_states is None else ambig_states

    # Filter points with an invalid state.
    valid = [p in invalid_states for p in points[-4, :]]
    points = points[:, valid]

    # Filter by dynProp.
    valid = [p in dynprop_states for p in points[3, :]]
    points = points[:, valid]

    # Filter by ambig_state.
    valid = [p in ambig_states for p in points[11, :]]
    points = points[:, valid]
    return points
