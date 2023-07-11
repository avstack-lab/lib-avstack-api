# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-09-05
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""
import logging
import os

import numpy as np

from avapi.nuscenes import nuScenesManager


nuScenes_data_dir = os.path.join(os.getcwd(), "data/nuScenes")
sensor_IDs = {
    "CAM_BACK": 0,
    "CAM_BACK_LEFT": 1,
    "CAM_BACK_RIGHT": 2,
    "CAM_FRONT": 3,
    "CAM_FRONT_LEFT": 4,
    "CAM_FRONT_RIGHT": 5,
    "LIDAR_TOP": 0,
    "RADAR_BACK_LEFT": 0,
    "RADAR_BACK_RIGHT": 1,
    "RADAR_FRONT": 2,
    "RADAR_FRONT_LEFT": 3,
    "RADAR_FRONT_RIGHT": 4,
}

if os.path.exists(os.path.join(nuScenes_data_dir, "v1.0-mini")):
    NSM = nuScenesManager(nuScenes_data_dir, "v1.0-mini")
    DM_0 = NSM.get_scene_dataset_by_index(0)
else:
    NSM = None
    DM_0 = None
    msg = "Cannot run test - nuScenes mini not downloaded"


def test_spawn_newscenes_dataset():
    if NSM is not None:
        assert len(DM_0.frames) == DM_0.scene["nbr_samples"]
    else:
        logging.warning(msg)
        print(msg)


def test_get_sensor_record():
    if NSM is not None:
        for sensor in sensor_IDs.keys():
            assert DM_0._get_sensor_record(0, sensor)["channel"] == sensor
    else:
        logging.warning(msg)
        print(msg)


def test_get_image():
    if NSM is not None:
        img = DM_0.get_image(0, "CAM_FRONT")
        rec = DM_0._get_sensor_record(0, "CAM_FRONT")
        assert img.shape == (rec["height"], rec["width"], 3)
    else:
        logging.warning(msg)
        print(msg)


def test_get_lidar():
    if NSM is not None:
        lid = DM_0.get_lidar(0, "LIDAR_TOP")
    else:
        logging.warning(msg)
        print(msg)


def test_camera_calibrations():
    if NSM is not None:
        calib = DM_0.get_calibration(0, "CAM_FRONT")
        assert calib.reference.x[0] > 0
        assert abs(calib.reference.x[1]) < 0.1
        assert calib.reference.x[2] > 0
        calib = DM_0.get_calibration(0, "CAM_BACK_RIGHT")
        assert calib.reference.x[0] > 0
        assert calib.reference.x[1] < 0
        assert calib.reference.x[2] > 0
        calib = DM_0.get_calibration(0, "CAM_BACK")
        assert abs(calib.reference.x[0]) < 0.2
        assert abs(calib.reference.x[1]) < 0.1
        assert calib.reference.x[2] > 0
    else:
        logging.warning(msg)
        print(msg)


def test_lidar_calibration():
    if NSM is not None:
        calib = DM_0.get_calibration(0, "LIDAR_TOP")
        assert calib.reference.x[0] > 0
        assert abs(calib.reference.x[1]) < 0.1
        assert calib.reference.x[2] > 0
    else:
        logging.warning(msg)
        print(msg)


def test_get_ego():
    if NSM is not None:
        ego = DM_0.get_ego(0)
        assert ego is not None
    else:
        logging.warning(msg)
        print(msg)


def test_get_objects():
    if NSM is not None:
        objects = DM_0.get_objects(0, "CAM_FRONT")
        assert len(objects) > 0
    else:
        logging.warning(msg)
        print(msg)
