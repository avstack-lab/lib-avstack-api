"""

"""

try:
    from nuimages import NuImages
except ModuleNotFoundError as e:
    print("Cannot import nuimages")

from .._nutonomy import _nuBaseDataset


class nuImagesDataset(_nuBaseDataset):
    NAME = "nuImages"
    CFG = {}
    CFG["IMAGE_WIDTH"] = 1600
    CFG["IMAGE_HEIGHT"] = 900
    CFG["IMAGE_CHANNEL"] = 3
    img_shape = (CFG["IMAGE_HEIGHT"], CFG["IMAGE_WIDTH"], CFG["IMAGE_CHANNEL"])
    sensors = {"main_camera": "CAM_FRONT", "main-camera": "CAM_FRONT"}
    keyframerate = 2
    sensor_IDs = {
        "CAM_BACK": 3,
        "CAM_BACK_LEFT": 1,
        "CAM_BACK_RIGHT": 2,
        "CAM_FRONT": 0,
        "CAM_FRONT_LEFT": 4,
        "CAM_FRONT_RIGHT": 5,
    }

    def __init__(
        self, data_dir, split, verbose=False, whitelist_types=..., ignore_types=...
    ):
        nuim = NuImages(version=split, dataroot=data_dir, verbose=verbose)
        self.t0 = 0
        self.ego_speed_interp = None
        super().__init__(
            nuim, None, data_dir, split, verbose, whitelist_types, ignore_types
        )

    def sensor_name(self, frame):
        return self.nuX.get(
            "sensor", self._get_calib_data(frame, None)["sensor_token"]
        )["channel"]

    def make_sample_records(self):
        self.sample_records = {i: v for i, v in enumerate(self.nuX.sample)}

    def get_image(self, frame, sensor):
        return super().get_image(frame, sensor)

    def get_calibration(self, frame, *args, **kwargs):
        return super().get_calibration(frame, None)

    def get_objects(self, frame, *args):
        return super().get_objects(frame, None)
