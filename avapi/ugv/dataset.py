import os

import numpy as np
from avstack.calibration import CameraCalibration, LidarCalibration, RadarCalibration
from avstack.config import DATASETS
from avstack.environment import ObjectState
from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position, Velocity
from avstack.geometry import transformations as tforms
from cv2 import imread

from .._dataset import BaseSceneDataset, BaseSceneManager


def get_splits_scenes(data_dir, seed=1, frac_train=0.7, frac_val=0.3):
    SM = BaseSceneManager(data_dir)
    return SM.make_splits_scenes(seed=seed, frac_train=frac_train, frac_val=frac_val)


# no objects yet in this dataset
_nominal_whitelist_types = []
_nominal_ignore_types = []


@DATASETS.register_module()
class UgvScenesManager(BaseSceneManager):
    name = "ugv"
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(
        self,
        data_dir: str,
        seed: int = 1,
        split_fracs={"train": 0.6, "val": 0.4, "test": 0.0},
        verbose: bool = False,
    ):
        if not os.path.exists(data_dir):
            raise RuntimeError(f"Cannot find data dir at {data_dir}")
        self.data_dir = data_dir
        self.verbose = verbose
        self.scenes = sorted(next(os.walk(data_dir))[1])
        self.splits_scenes = self.make_splits_scenes(
            seed=seed,
            frac_train=split_fracs["train"],
            frac_val=split_fracs["val"],
            frac_test=split_fracs["test"],
        )

    def get_scene_dataset_by_index(self, scene_idx):
        return UgvSceneDataset(self.data_dir, self.scenes[scene_idx])

    def get_scene_dataset_by_name(self, scene_name):
        if not scene_name in self.scenes:
            raise IndexError(f"Cannot find scene {scene_name} in {self.scenes}")
        return UgvSceneDataset(self.data_dir, scene_name)


class UgvSceneDataset(BaseSceneDataset):
    name = "ugv"
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(
        self,
        data_dir: str,
        scene: str,
        whitelist_types: list = _nominal_whitelist_types,
        ignore_types: list = _nominal_ignore_types,
        radar_folder: str = "radar_0",
        lidar_folder: str = "lidar",
        camera_folder: str = "camera",
        imu_orientation_folder: str = "imu_data",
        imu_full_folder: str = "imu_data_full",
        vehicle_vel_folder: str = "vehicle_vel",
    ):

        # avstack fields
        self.data_dir = data_dir
        self.scene = scene
        self.sequence_id = scene
        self.sensor_IDs = {
            "radar": radar_folder,
            "lidar": lidar_folder,
            "camera": camera_folder,
            "orientation": imu_orientation_folder,
            "imu": imu_full_folder,
        }

        self.sensors = self.sensor_IDs
        self.scene_path = os.path.join(data_dir, scene)
        super().__init__(whitelist_types, ignore_types)

        # radar data
        self.radar_enabled = False
        self.radar_folder = radar_folder
        self.radar_files = []

        # lidar data
        self.lidar_enabled = False
        self.lidar_folder = lidar_folder
        self.lidar_files = []

        # camera data
        self.camera_enabled = False
        self.camera_folder = camera_folder
        self.camera_files = []
        self._camera_P = np.ones((3, 4))  # HACK
        self._img_shape = (480, 640)

        # imu data - orientation only
        self.imu_orientation_folder = imu_orientation_folder
        self.imu_orientation_files = []
        self.imu_orientation_enabled = False

        # imu data - full sensor data
        self.imu_full_folder = imu_full_folder
        self.imu_full_files = []
        self.imu_full_enabled = False

        # vehicle velocity data
        self.vehicle_vel_folder = vehicle_vel_folder
        self.vehicle_vel_files = []
        self.vehicle_vel_enabled = False

        # variable to keep track of the number of frames
        self.num_frames = 0

        # load the new dataset
        self.load_new_dataset(self.scene_path)
        self.frames = list(range(self.num_frames))

    def __str__(self):
        return f"UGV Dataset of folder: {self.scene_path}"

    def _load_calibration(
        self, frame: int, sensor: int, agent: int = None, *args, **kwargs
    ):
        # -- get the sensor reference: HACK - assume for now platform is origin
        reference = GlobalOrigin3D

        # -- get the calibration
        if "radar" in sensor.lower():
            fov_horz = 180 * np.pi / 180  # HACK
            fov_vert = 30 * np.pi / 180  # HACK
            calib = RadarCalibration(reference, fov_horz, fov_vert)
        elif "lidar" in sensor.lower():
            calib = LidarCalibration(reference)
        elif "cam" in sensor.lower():
            calib = CameraCalibration(
                reference=reference,
                P=self._camera_P,
                img_shape=self._img_shape,
            )
        else:
            raise NotImplementedError(sensor)

        return calib

    def get_agents(self, frame: int):
        return [self._load_ego(frame=frame)]

    def get_ego_reference(self, *args, **kwargs):
        return None

    def _load_agent_set(self, frame: int) -> set:
        # TODO: this is slow...improve
        return {ag.ID for ag in self.get_agents(frame)}

    def _load_ego(self, frame: int, agent: int = None):
        ts = self.get_timestamp(frame=frame, sensor=None)
        obj = ObjectState(obj_type="ground_vehicle", ID=1000)

        # load data
        heading = self.get_imu_orientation_rad(frame)
        vel = [*self.get_vehicle_vel_data(frame)[0, 1:3], 0]  # select first index

        # make data classes
        position = Position(x=np.zeros((3,)), reference=GlobalOrigin3D)  # HACK
        velocity = Velocity(x=vel, reference=GlobalOrigin3D)
        attitude = Attitude(
            q=tforms.transform_orientation([0, 0, heading], "euler", "quat"),
            reference=GlobalOrigin3D,
        )  # HACK
        box = Box3D(position=position, attitude=attitude, hwl=[1, 1, 1])
        obj.set(
            t=ts,
            position=position,
            velocity=velocity,
            box=box,
        )
        return obj

    def _load_frames(self, sensor: str, agent: int):
        return list(range(self.num_frames))

    def _load_timestamp(self, frame: int, sensor: str = None, *args, **kwargs):
        return self.get_imu_full_data(frame)[0][0]  # first slot is time

    def _load_radar(self, frame: int, sensor: int, agent: int = None):
        return self.get_radar_data(frame)

    def _load_lidar(
        self,
        frame: int,
        sensor: int,
        agent: int = None,
        filter_front: bool = False,
        *args,
        **kwargs,
    ):
        pcd = self.get_lidar_point_cloud_raw(frame)[:, :3]  # just positions
        if filter_front:
            return pcd[pcd[:, 0] > 0, :]  # assumes z is forward....
        else:
            return pcd

    def _load_image(self, frame: int, sensor: int, agent: int = None):
        img = self.get_camera_frame(frame)
        return img

    def _load_objects(self, *args, **kwargs):
        return []

    #########################################################
    # David's methods
    #########################################################

    def load_new_dataset(self, dataset_path: str):

        self.dataset_path = dataset_path
        self.import_dataset_files()
        self.determine_num_frames()

    def import_dataset_files(self):

        self.import_radar_data()
        self.import_lidar_data()
        self.import_camera_data()
        self.import_imu_orientation_data()
        self.import_imu_full_data()
        self.import_vehicle_vel_data()

    def determine_num_frames(self):

        self.num_frames = 0

        if self.radar_enabled:
            self.set_num_frames(len(self.radar_files))
        if self.lidar_enabled:
            self.set_num_frames(len(self.lidar_files))
        if self.camera_enabled:
            self.set_num_frames(len(self.camera_files))
        if self.imu_full_enabled:
            self.set_num_frames(len(self.imu_full_files))
        if self.imu_orientation_enabled:
            self.set_num_frames(len(self.imu_orientation_files))
        if self.vehicle_vel_enabled:
            self.set_num_frames(len(self.vehicle_vel_files))

        return

    def set_num_frames(self, num_files: int):
        """Update the number of frames available in the dataset

        Args:
            num_files (int): The number of files available for a given sensor
        """
        if self.num_frames > 0:
            self.num_frames = min(self.num_frames, num_files)
        else:
            self.num_frames = num_files

    ####################################################################
    # handling radar data
    ####################################################################
    def import_radar_data(self):

        path = os.path.join(self.dataset_path, self.radar_folder)

        if os.path.isdir(path):
            self.radar_enabled = True
            self.radar_files = sorted(os.listdir(path))
            print("found {} radar samples".format(len(self.radar_files)))
        else:
            print("did not find radar samples")

        return

    def get_radar_data(self, idx: int) -> np.ndarray:
        """Get radar detections or ADC data cube for a specific index in the dataset

        Args:
            idx (int): The index of the radar data detection

        Returns:
            np.ndarray: An Nx4 of radar detections with (x,y,z,vel) vals or
                        (rx_channels) x (samples) x (chirps) ADC cube for a given frame
        """

        assert self.radar_enabled, "No radar dataset loaded"

        path = os.path.join(self.dataset_path, self.radar_folder, self.radar_files[idx])

        points = np.load(path)

        return points

    ####################################################################
    # handling lidar data
    ####################################################################
    def import_lidar_data(self):

        path = os.path.join(self.dataset_path, self.lidar_folder)

        if os.path.isdir(path):
            self.lidar_enabled = True
            self.lidar_files = sorted(os.listdir(path))
            print("found {} lidar samples".format(len(self.lidar_files)))
        else:
            print("did not find lidar samples")

        return

    def get_lidar_point_cloud(self, idx) -> np.ndarray:
        path = os.path.join(self.dataset_path, self.lidar_folder, self.lidar_files[idx])
        """Get a lidar pointcloud from the desired frame,
        filters out ground and higher detections points

        Returns:
            np.ndarray: a Nx2 array of lidar detections
        """
        assert self.lidar_enabled, "No lidar dataset loaded"
        points = np.load(path)

        valid_points = points[:, 2] > -0.2  # filter out ground
        valid_points = valid_points & (points[:, 2] < 0.1)  # higher elevation points

        points = points[valid_points, :2]

        return points

    def get_lidar_point_cloud_raw(self, idx) -> np.ndarray:
        path = os.path.join(self.dataset_path, self.lidar_folder, self.lidar_files[idx])
        """Get a lidar pointcloud from the desired frame,
        without filtering anything out

        Returns:
            np.ndarray: a Nx3 array of lidar detections
        """
        assert self.lidar_enabled, "No lidar dataset loaded"
        points = np.load(path)

        return points

    ####################################################################
    # handling camera data
    ####################################################################
    def import_camera_data(self):

        path = os.path.join(self.dataset_path, self.camera_folder)

        if os.path.isdir(path):
            self.camera_enabled = True
            self.camera_files = sorted(os.listdir(path))
            print("found {} camera samples".format(len(self.camera_files)))
        else:
            print("did not find camera samples")

        return

    def get_camera_frame(self, idx: int) -> np.ndarray:
        """Get a camera frame from the dataset

        Args:
            idx (int): the index in the dataset to get the camera
                data from

        Returns:
            np.ndarray: the camera data with rgb channels
        """
        assert self.camera_enabled, "No camera dataset loaded"

        path = os.path.join(
            self.dataset_path, self.camera_folder, self.camera_files[idx]
        )
        image = imread(path)

        # return while also flipping red and blue channel
        return image[:, :, ::-1]

    ####################################################################
    # handling imu data (orientation only)
    ####################################################################
    def import_imu_orientation_data(self):

        path = os.path.join(self.dataset_path, self.imu_orientation_folder)

        if os.path.isdir(path):
            self.imu_orientation_enabled = True
            self.imu_orientation_files = sorted(os.listdir(path))
            print(
                "found {} imu (orientation only) samples".format(
                    len(self.imu_orientation_files)
                )
            )
        else:
            print("did not find imu (orientation) samples")

        return

    def get_imu_orientation_rad(self, idx: int):
        """Get the raw imu heading from the dataset at a given frame index

        Args:
            idx (int): the frame index to get the imu heading for

        Returns:
            _type_: the raw heading read from the IMU expressed in the range
                [-pi,pi]
        """
        assert self.imu_orientation_enabled, "No IMU (orientation) dataset loaded"

        path = os.path.join(
            self.dataset_path,
            self.imu_orientation_folder,
            self.imu_orientation_files[idx],
        )

        data = np.load(path)
        w = data[0]
        x = data[1]
        y = data[2]
        z = data[3]

        heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return heading

    ####################################################################
    # handling imu (full sensor) data
    ####################################################################
    def import_imu_full_data(self):

        path = os.path.join(self.dataset_path, self.imu_full_folder)

        if os.path.isdir(path):
            self.imu_full_enabled = True
            self.imu_full_files = sorted(os.listdir(path))
            print("found {}imu (full data) samples".format(len(self.imu_full_files)))
        else:
            print("did not find imu (full data) samples")

        return

    def get_imu_full_data(self, idx=0):
        """_summary_

        Args:
            idx (int, optional): _description_. Defaults to 0.

        Returns:
            np.ndarray: [time,w_x,w_y,w_z,acc_x,acc_y,acc_z]
        """
        assert self.imu_full_enabled, "No IMU Full dataset loaded"

        # load the data sample
        path = os.path.join(
            self.dataset_path, self.imu_full_folder, self.imu_full_files[idx]
        )

        return np.load(path)

    ####################################################################
    # handling vehicle velocity data
    ####################################################################
    def import_vehicle_vel_data(self):

        path = os.path.join(self.dataset_path, self.vehicle_vel_folder)

        if os.path.isdir(path):
            self.vehicle_vel_enabled = True
            self.vehicle_vel_files = sorted(os.listdir(path))
            print(
                "found {} vehicle velocity samples".format(len(self.vehicle_vel_files))
            )
        else:
            print("did not find vehicle velocity samples")

        return

    def get_vehicle_vel_data(self, idx=0):

        assert self.vehicle_vel_files, "No Vehicle velocity dataset loaded"

        # load the data sample
        path = os.path.join(
            self.dataset_path, self.vehicle_vel_folder, self.vehicle_vel_files[idx]
        )

        return np.load(path)
