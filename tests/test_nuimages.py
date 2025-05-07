"""

"""

import logging
import os

from avapi.nutonomy import nuImagesDataset


nuImages_data_dir = os.path.join(os.getcwd(), "data/nuImages")
if os.path.exists(os.path.join(nuImages_data_dir, "v1.0-mini")):
    NID = nuImagesDataset(nuImages_data_dir, "v1.0-mini")
else:
    NID = None
    msg = "Cannot run test - nuImages mini not downloaded"


def test_init_newimages_dataset():
    if NID is not None:
        assert len(NID.frames) == len(NID.nuX.sample)
    else:
        logging.warning(msg)
        print(msg)


def test_get_sensor_record():
    if NID is not None:
        sr = NID._get_sensor_record(0)
        assert isinstance(sr, dict)
    else:
        logging.warning(msg)
        print(msg)


def test_get_image():
    if NID is not None:
        img = NID.get_image(0)
        rec = NID._get_sensor_record(0)
        assert img.shape == (rec["height"], rec["width"], 3)
    else:
        logging.warning(msg)


def test_camera_calibrations():
    if NID is not None:
        frame = 2
        assert NID.sensor_name(frame) == "CAM_BACK"
        calib = NID.get_calibration(frame)
        assert abs(calib.reference.x[0]) < 0.2
        assert abs(calib.reference.x[1]) < 0.1
        assert calib.reference.x[2] > 0
    else:
        logging.warning(msg)
        print(msg)


def test_get_ego():
    if NID is not None:
        ego = NID.get_ego(0)
        assert ego is not None
    else:
        logging.warning(msg)
        print(msg)
