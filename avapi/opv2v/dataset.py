"""
OPV2V dataset manager on AVstack conventions
"""

import glob
import os

import numpy as np
import open3d as o3d
import yaml
from avstack import calibration
from avstack.environment.objects import Occlusion, VehicleState
from avstack.geometry import (
    ReferenceFrame,
    Rotation,
    Vector,
    bbox,
    q_mult_vec,
    q_stan_to_cam,
)
from avstack.geometry import transformations as tforms
from cv2 import imread, imwrite

from .._dataset import BaseSceneDataset, BaseSceneManager


_nominal_whitelist_types = ["car", "bicycle", "truck", "motorcycle"]
_nominal_ignore_types = []


class Opv2vScenesManager(BaseSceneManager):
    name = "OPV2V"

    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(self, data_dir, split="train", verbose=False):
        """
        data_dir: the base folder where all scenes are kept
        """
        raise NotImplementedError('The OPV2V scene manager will not work'
                    ' until someone updates it to the latest AVstack'
                    ' geometry format.')
        if not os.path.exists(data_dir):
            raise RuntimeError(f"Cannot find data dir at {data_dir}")
        self.split = "train"
        self.data_dir = os.path.join(data_dir, split)
        self.scenes_paths = {}
        self.agent_paths = {}
        root, subs, _ = next(os.walk(self.data_dir))
        subs = sorted(subs)
        for iscene, sub in enumerate(subs):
            self.scenes_paths[iscene] = os.path.join(root, sub)
            self.agent_paths[iscene] = {}
            subs2 = sorted(next(os.walk(os.path.join(root, sub)))[1])
            for iagent, sub2 in enumerate(subs2):
                self.agent_paths[iscene][iagent] = os.path.join(root, sub, sub2)

    def list_agents(self, scene_idx):
        print(self.agent_paths[scene_idx].keys())

    def get_scene_datset_by_index(self, scene_idx, agent_idx):
        return Opv2vSceneDataset(
            self.data_dir, scene_idx, agent_idx, self.agent_paths[scene_idx][agent_idx]
        )


class Opv2vSceneDataset(BaseSceneDataset):
    name = "OPV2V"
    CFG = {}
    CFG["num_lidar_features"] = 3
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types
    sensors = {
        "main_lidar": "lidar",
        "lidar": "lidar",
        "main_camera": "camera0",
    }
    framerate = 10  # need to confirm this
    img_shape = (600, 800)
    sensor_IDs = {"camera0": 0, "camera1": 1, "camera2": 2, "camera3": 3, "lidar": 0}

    def __init__(
        self,
        data_dir,
        scene_idx,
        agent_idx,
        scene_path,
        whitelist_types=_nominal_whitelist_types,
        ignore_types=_nominal_ignore_types,
    ):
        raise NotImplementedError('The OPV2V scene manager will not work'
            ' until someone updates it to the latest AVstack'
            ' geometry format.')
        self.data_dir = data_dir
        self.scene = scene_idx
        self.agent = agent_idx
        self.sequence_id = (
            1019 * scene_idx + agent_idx
        )  # is this unique given small agent numbers?
        self.scene_path = os.path.join(data_dir, scene_path)

        # prep scene information
        self.frames = []
        for file in sorted(glob.glob(os.path.join(self.scene_path, "*.yaml"))):
            self.frames.append(int(file.split("/")[-1].replace(".yaml", "")))
        super().__init__(whitelist_types, ignore_types)

    def __str__(self):
        return f"OPV2V dataset of folder: {self.scene_path}"

    def check_frame(self, frame):
        assert (
            frame in self.frames
        ), f"Candidate frame, {frame}, not in frame set {self.frames}"

    def _load_yaml(self, frame):
        self.check_frame(frame)
        with open(os.path.join(self.scene_path, "%06d.yaml" % frame), "r") as stream:
            try:
                yam = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc
        return yam

    def _load_calibration(self, frame, sensor):
        yam = self._load_yaml(frame)
        if "camera" in sensor.lower():
            # Orientation
            B_ryp = yam["true_ego_pos"][3:6]
            B_rpy = [
                np.pi / 180 * B_ryp[0],
                np.pi / 180 * -B_ryp[2],
                np.pi / 180 * -B_ryp[1],
            ]
            C_ryp = yam[sensor]["cords"][3:6]
            C_rpy = [C_ryp[0], -C_ryp[2], -C_ryp[1]]
            C_rpy = [
                np.pi / 180 * C_ryp[0],
                np.pi / 180 * -C_ryp[2],
                np.pi / 180 * -C_ryp[1],
            ]  # negative bc carla
            q_O_2_C = transform_orientation(C_rpy, "euler", "quat")
            q_O_2_B = transform_orientation(B_rpy, "euler", "quat")
            q_B_2_C = q_stan_to_cam * q_O_2_C * q_O_2_B.conjugate()

            # Position
            x_O_2_B_in_O = np.array(yam["true_ego_pos"][:3])
            x_O_2_B_in_O[1] *= -1
            x_O_2_C_in_O = np.array(yam[sensor]["cords"][:3])
            x_O_2_C_in_O[1] *= -1
            x_B_2_C_in_O = x_O_2_C_in_O - x_O_2_B_in_O
            x_B_2_C_in_B = q_mult_vec(q_O_2_B, x_B_2_C_in_O)

            # Intrinsics
            P = np.zeros((3, 4))
            P[:3, :3] = np.array(yam[sensor]["intrinsic"])

            # Calibration
            origin = Origin(x_B_2_C_in_B, q_B_2_C)
            calib = calibration.CameraCalibration(origin, P, self.img_shape)
        elif "lidar" in sensor.lower():
            # Orientation
            B_ryp = yam["true_ego_pos"][3:6]
            B_rpy = [
                np.pi / 180 * B_ryp[0],
                np.pi / 180 * -B_ryp[2],
                np.pi / 180 * -B_ryp[1],
            ]  # negative bc carla
            L_ryp = yam["lidar_pose"][3:6]
            L_rpy = [
                np.pi / 180 * L_ryp[0],
                np.pi / 180 * -L_ryp[2],
                np.pi / 180 * -L_ryp[1],
            ]  # negative bc carla
            q_O_2_B = transform_orientation(B_rpy, "euler", "quat")
            q_O_2_L = transform_orientation(L_rpy, "euler", "quat")
            q_B_2_L = q_O_2_L * q_O_2_B.conjugate()

            # Position
            x_O_2_B_in_O = np.array(yam["true_ego_pos"][:3])
            x_O_2_B_in_O[1] *= -1
            x_O_2_L_in_O = np.array(yam["lidar_pose"][:3])
            x_O_2_L_in_O[1] *= -1
            x_B_2_L_in_O = x_O_2_L_in_O - x_O_2_B_in_O
            x_B_2_L_in_B = q_mult_vec(q_O_2_B, x_B_2_L_in_O)

            # Calibration
            calib = calibration.Calibration(Origin(x_B_2_L_in_B, q_B_2_L))
        else:
            raise NotImplementedError(sensor)
        return calib

    def _load_image(self, frame, sensor):
        self.check_frame(frame)
        if isinstance(sensor, str) and "camera" in sensor:
            img_fname = os.path.join(self.scene_path, "%06d_%s.png" % (frame, sensor))
        else:
            img_fname = os.path.join(
                self.scene_path, "%06d_camera%d.png" % (frame, sensor)
            )
        return imread(img_fname)[:, :, ::-1]

    def _load_lidar(self, frame, sensor, filter_front=False, **kwargs):
        pcd = o3d.io.read_point_cloud(os.path.join(self.scene_path, "%06d.pcd" % frame))
        lidar = np.asarray(pcd.points)
        lidar[:, 1] *= -1
        if filter_front:
            return lidar[lidar[:, 0] > 0, :]
        else:
            return lidar

    def _load_objects(
        self,
        frame,
        sensor,
        whitelist_types=_nominal_whitelist_types,
        ignore_types=_nominal_ignore_types,
    ):
        object_calib = self.get_calibration(frame, sensor)
        yam = self._load_yaml(frame)
        objects = []
        for ID, vehicle in yam["vehicles"].items():
            object_origin = NominalOriginStandard

            # Orientation
            B_ryp = yam["true_ego_pos"][3:6]
            B_rpy = [
                np.pi / 180 * B_ryp[0],
                np.pi / 180 * -B_ryp[2],
                np.pi / 180 * -B_ryp[1],
            ]  # negative bc carla
            V_ryp = vehicle["angle"]
            V_rpy = [
                np.pi / 180 * V_ryp[0],
                np.pi / 180 * -V_ryp[2],
                np.pi / 180 * -V_ryp[1],
            ]  # negative bc carla
            q_O_2_B = transform_orientation(B_rpy, "euler", "quat")
            q_O_2_V = transform_orientation(V_rpy, "euler", "quat")
            q_B_2_V = q_O_2_V * q_O_2_B.conjugate()

            # Position
            x_O_2_B_in_O = np.array(yam["true_ego_pos"][:3])
            x_O_2_B_in_O[1] *= -1
            x_O_2_V_in_O = np.array(vehicle["location"])
            x_O_2_V_in_O[1] *= -1
            x_B_2_V_in_O = x_O_2_V_in_O - x_O_2_B_in_O
            x_B_2_V_in_B = q_mult_vec(q_O_2_B, x_B_2_V_in_O)

            # Velocity
            v_body_B = np.array([yam["ego_speed"], 0, 0]) * 1000 / 3600
            v_body_V = np.array([vehicle["speed"], 0, 0]) * 1000 / 3600
            v_forward_V = q_mult_vec(q_B_2_V.conjugate(), v_body_V)  # TODO: check this
            v_forward_B = v_body_B
            v_delta = v_forward_V - v_forward_B

            # Bounding box
            l, w, h = [2 * e for e in vehicle["extent"]]

            # Wrap in vehicle state
            obj = VehicleState("car", ID)
            pos = x_B_2_V_in_B
            box3d = bbox.Box3D(
                [h, w, l, x_B_2_V_in_B, q_B_2_V], object_origin, where_is_t="bottom"
            )
            vel = v_delta
            acc = None
            att = q_B_2_V
            ang = None
            occ = Occlusion.UNKNOWN
            obj.set(
                float(self.get_timestamp(frame, sensor)),
                pos,
                box3d,
                vel,
                acc,
                att,
                ang,
                occlusion=occ,
                origin=object_origin,
            )
            obj.change_origin(object_calib.origin)
            objects.append(obj)
        return objects

    def _load_timestamp(self, frame, sensor):
        return self.framerate * frame

    def _save_objects(self, frame, sensor, folder, file):
        raise NotImplementedError
