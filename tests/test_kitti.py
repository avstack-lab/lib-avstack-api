# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-19
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-08
# @Description:
"""

"""
import logging
import os

from avapi.kitti import KittiObjectDataset as KOD
from avapi.kitti import KittiRawDataset as KRD


# KITTI object
KITTI_obj_data_dir = os.path.join(os.getcwd(), "data/KITTI/object")
if os.path.exists(os.path.join(KITTI_obj_data_dir, "training")):
    DM_obj = KOD(KITTI_obj_data_dir, "training")
else:
    DM_obj = None
    msg_obj = "Cannot run test - KITTI object data not downloaded"

# KITTI raw
KITTI_raw_data_dir = os.path.join(os.getcwd(), "data/KITTI/raw")
if os.path.exists(KITTI_raw_data_dir):
    DM_raw = KRD(KITTI_raw_data_dir)
else:
    DM_raw = None
    msg_raw = "Cannot run test - KITTI raw data not downloaded"

"""
Kitti Object Dataset
"""


def test_get_image():
    if DM_obj is not None:
        img = DM_obj.get_image(1, sensor=2)
        assert img.frame == 1
        assert img.shape == (
            DM_obj.CFG["IMAGE_HEIGHT"],
            DM_obj.CFG["IMAGE_WIDTH"],
            DM_obj.CFG["IMAGE_CHANNEL"],
        )
    else:
        logging.warning(msg_obj)
        print(msg_obj)


def test_get_lidar():
    if DM_obj is not None:
        pc = DM_obj.get_lidar(1)
        assert pc.frame == 1
        assert pc.shape[1] == DM_obj.CFG["num_lidar_features"]
    else:
        logging.warning(msg_obj)
        print(msg_obj)


def test_get_objects():
    if DM_obj is not None:
        objs = DM_obj.get_objects(1)
        assert len(objs) == 2
    else:
        logging.warning(msg_obj)
        print(msg_obj)


"""
Kitti Raw Dataset
"""


def test_init_krd():
    if DM_raw is not None:
        dates = DM_raw.get_available_dates()
        assert "2011_09_26" in dates
        sequence_ids = DM_raw.get_sequence_ids_at_date(dates[0], True)
        assert "2011_09_26_drive_0001_sync" in sequence_ids
    else:
        logging.warning(msg_raw)
        print(msg_raw)


def test_convert_krd():
    if DM_raw is not None:
        dates = DM_raw.get_available_dates()
        exp_path = DM_raw.convert_sequence(
            dates[0],
            idx_seq=0,
            max_frames=None,
            max_time=None,
            tracklets_req=True,
            path_append="-experiment",
        )
        DM = KOD(KITTI_obj_data_dir, exp_path)
        assert len(DM.frames) > 0
    else:
        logging.warning(msg_raw)
        print(msg_raw)
