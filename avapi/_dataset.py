# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-08-08
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-09-29
# @Description:
"""

"""

import json
import os
import random

import numpy as np
from avstack import calibration, sensors
from avstack.environment.objects import ObjectStateDecoder, Occlusion, VehicleState
from avstack.geometry import (
    Acceleration,
    AngularVelocity,
    Attitude,
    Box3D,
    GlobalOrigin3D,
    PointMatrix3D,
    Position,
    ReferenceFrame,
    Velocity,
    q_cam_to_stan,
    q_mult_vec,
)
from avstack.geometry import transformations as tforms
from avstack.modules.tracking.tracks import TrackContainerDecoder
from cv2 import imread


def wrap_minus_pi_to_pi(phases):
    phases = (phases + np.pi) % (2 * np.pi) - np.pi
    return phases


class BaseSceneManager:
    def __iter__(self):
        for scene in self.scenes:
            yield self.get_scene_dataset_by_name(scene)

    def __len__(self):
        return len(self.scenes)

    @property
    def name(self):
        return self.NAME

    def list_scenes(self):
        print(self.scenes)

    def get_splits_scenes(self):
        return self.splits_scenes

    def make_splits_scenes(self, seed=1, frac_train=0.7, frac_val=0.3):
        """Split the scenes by hashing the experiment name and modding
        3:1 split using mod 4
        """
        rng = random.Random(seed)

        # first two we alternate just to have one
        splits_scenes = {"train": [], "val": [], "test": []}
        for i, scene in enumerate(self.scenes):
            if i == 0:
                splits_scenes["train"].append(scene)
            elif i == 1:
                splits_scenes["val"].append(scene)
            else:
                rv = rng.random()
                if rv < frac_train:
                    splits_scenes["train"].append(scene)
                elif rv < (frac_train + frac_val):
                    splits_scenes["val"].append(scene)
                else:
                    splits_scenes["test"].append(scene)
        return splits_scenes

    def get_scene_dataset_by_name(self):
        raise NotImplementedError

    def get_scene_dataset_by_index(self):
        raise NotImplementedError


class BaseSceneDataset:
    sensor_IDs = {}

    def __init__(self, whitelist_types, ignore_types):
        self.whitelist_types = whitelist_types
        self.ignore_types = ignore_types

    def __str__(self):
        return f"{self.NAME} dataset of folder: {self.split_path}"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.frames)

    @property
    def name(self):
        return self.NAME

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, frames):
        self._frames = frames

    def check_frame(self, frame):
        assert (
            frame in self.frames
        ), f"Candidate frame, {frame}, not in frame set {self.frames}"

    def get_sensor_ID(self, sensor):
        try:
            return self.sensor_IDs[sensor]
        except KeyError as e:
            try:
                return self.sensor_IDs[self.sensors[sensor]]
            except KeyError as e2:
                raise e2

    def get_sensor_name(self, sensor):
        if (sensor is None) or (sensor not in self.sensors):
            return sensor
        else:
            return self.sensors[sensor]

    def get_frames(self, sensor):
        sensor = self.get_sensor_name(sensor)
        return self._load_frames(sensor)

    def get_calibration(self, frame, sensor):
        sensor = self.get_sensor_name(sensor)
        ego_reference = self.get_ego_reference(frame)
        return self._load_calibration(frame, sensor=sensor, ego_reference=ego_reference)

    def get_ego(self, frame):
        return self._load_ego(frame)

    def get_ego_reference(self, frame):
        return self.get_ego(frame).as_reference()

    def get_sensor_data_filepath(self, frame, sensor):
        sensor = self.get_sensor_name(sensor)
        return self._load_sensor_data_filepath(frame, sensor)

    def get_image(self, frame, sensor=None):
        sensor = self.get_sensor_name(sensor)
        ts = self.get_timestamp(frame, sensor)
        data = self._load_image(frame, sensor=sensor)
        cam_string = "image-%i" % sensor if isinstance(sensor, int) else sensor
        cam_string = (
            cam_string if sensor is not None else self.get_sensor_name("main_camera")
        )
        calib = self.get_calibration(frame, cam_string)
        return sensors.ImageData(
            ts, frame, data, calib, self.get_sensor_ID(cam_string), channel_order="rgb"
        )

    def get_semseg_image(self, frame, sensor=None):
        if sensor is None:
            sensor = self.sensors["semseg"]
        sensor = self.get_sensor_name(sensor)
        ts = self.get_timestamp(frame, sensor)
        data = self._load_semseg_image(frame, sensor=sensor)
        cam_string = "image-%i" % sensor if isinstance(sensor, int) else sensor
        calib = self.get_calibration(frame, cam_string)
        return sensors.SemanticSegmentationImageData(
            ts, frame, data, calib, self.get_sensor_ID(cam_string)
        )

    def get_depth_image(self, frame, sensor=None):
        if sensor is None:
            sensor = self.sensors["depth"]
        sensor = self.get_sensor_name(sensor)
        ts = self.get_timestamp(frame, sensor)
        data = self._load_depth_image(frame, sensor=sensor)
        cam_string = "image-%i" % sensor if isinstance(sensor, int) else sensor
        calib = self.get_calibration(frame, cam_string)
        return sensors.DepthImageData(
            ts, frame, data, calib, self.get_sensor_ID(cam_string)
        )

    def get_lidar(
        self,
        frame,
        sensor=None,
        filter_front=False,
        min_range=None,
        max_range=None,
        with_panoptic=False,
    ):
        if sensor is None:
            sensor = self.sensors["lidar"]
        sensor = self.get_sensor_name(sensor)
        ts = self.get_timestamp(frame, sensor)
        calib = self.get_calibration(frame, sensor)
        data = self._load_lidar(
            frame, sensor, filter_front=filter_front, with_panoptic=with_panoptic
        )
        data = PointMatrix3D(data, calib)
        pc = sensors.LidarData(ts, frame, data, calib, self.get_sensor_ID(sensor))
        if (min_range is not None) or (max_range is not None):
            pc.filter_by_range(min_range, max_range, inplace=True)
        return pc

    def get_radar(
        self,
        frame,
        sensor=None,
        min_range=None,
        max_range=None,
    ):
        if sensor is None:
            sensor = self.sensors["radar"]
        sensor = self.get_sensor_name(sensor)
        ts = self.get_timestamp(frame, sensor)
        calib = self.get_calibration(frame, sensor)
        data = self._load_radar(frame, sensor)  # razelrrt data
        data = PointMatrix3D(data, calib)
        rad = sensors.RadarDataRazelRRT(
            ts, frame, data, calib, self.get_sensor_ID(sensor)
        )
        if (min_range is not None) or (max_range is not None):
            rad.filter_by_range(min_range, max_range, inplace=True)
        return rad

    def get_objects(
        self, frame, sensor="main_camera", max_dist=None, max_occ=None, **kwargs
    ):
        reference = self.get_ego_reference(frame)
        sensor = self.get_sensor_name(sensor)
        objs = self._load_objects(frame, sensor=sensor, **kwargs)
        if max_occ is not None:
            objs = np.array(
                [
                    obj
                    for obj in objs
                    if (obj.occlusion <= max_occ)
                    or (obj.occlusion == Occlusion.UNKNOWN)
                ]
            )
        if max_dist is not None:
            if sensor == "ego":
                calib = calibration.Calibration(reference)
            else:
                calib = self.get_calibration(frame, sensor)
            objs = np.array(
                [
                    obj
                    for obj in objs
                    if obj.position.distance(calib.reference) < max_dist
                ]
            )
        return objs

    def get_objects_global(self, frame, max_dist=None, **kwargs):
        return self._load_objects_global(frame, **kwargs)

    def get_number_of_objects(self, frame, **kwargs):
        return self._number_objects_from_file(frame, **kwargs)

    def get_objects_from_file(self, fname, whitelist_types, max_dist=None):
        return self._load_objects_from_file(fname, whitelist_types, max_dist=max_dist)

    def get_timestamp(self, frame, sensor="lidar", utime=False):
        sensor = self.get_sensor_name(sensor)
        return self._load_timestamp(frame, sensor=sensor, utime=utime)

    def get_sensor_file_path(self, frame, sensor):
        return self._get_sensor_file_name(frame, sensor)

    def save_calibration(self, frame, calib, folder, **kwargs):
        if not os.path.exists(folder):
            os.makedirs(folder)
        self._save_calibration(frame, calib, folder, **kwargs)

    def save_objects(self, frame, objects, folder, file=None):
        if not os.path.exists(folder):
            os.makedirs(folder)
        self._save_objects(frame, objects, folder, file=file)

    def _load_frames(self, sensor):
        raise NotImplementedError

    def _load_calibration(self, frame, sensor, reference):
        raise NotImplementedError

    def _load_image(self, frame, camera):
        raise NotImplementedError

    def _load_semseg_image(self, frame, camera):
        raise NotImplementedError

    def _load_depth_image(self, frame, camera):
        raise NotImplementedError

    def _load_lidar(self, frame):
        raise NotImplementedError

    def _load_objects(self, frame, sensor):
        raise NotImplementedError

    def _load_sensor_data_filepath(self, frame, sensor: str):
        return self._get_sensor_file_name(frame, sensor)

    def _load_objects_from_file(
        self,
        fname,
        whitelist_types=None,
        ignore_types=None,
        max_dist=None,
        dist_ref=GlobalOrigin3D,
    ):
        # -- prep whitelist types
        if whitelist_types is None:
            whitelist_types = self.nominal_whitelist_types
        if not isinstance(whitelist_types, list):
            whitelist_types = [whitelist_types]
        whitelist_types = [wh.lower() for wh in whitelist_types]
        # -- prep ignore types
        if ignore_types is None:
            ignore_types = self.nominal_ignore_types
        if not isinstance(ignore_types, list):
            ignore_types = [ignore_types]
        ignore_types = [ig.lower() for ig in ignore_types]
        # -- read objects
        with open(fname, "r") as f:
            lines = [line.rstrip() for line in f.readlines()]
        objects = []
        for line in lines:
            if "datacontainer" in line:
                dc = json.loads(line, cls=TrackContainerDecoder)
                objects = dc.data
                break
            else:
                obj = self.parse_label_line(line)
            # -- type filter
            if ("all" in whitelist_types) or (obj.obj_type.lower() in whitelist_types):
                if obj.obj_type.lower() in ignore_types:
                    continue
            else:
                continue
            # -- distance filter
            if max_dist is not None:
                obj.change_reference(dist_ref, inplace=True)
                if obj.position.norm() > max_dist:
                    continue
            # -- save if made to here
            objects.append(obj)

        return np.array(objects)

    def _load_timestamp(self, frame):
        raise NotImplementedError

    def _save_objects(self, frame, objects, folder):
        raise NotImplementedError

    @staticmethod
    def read_dict_text_file(filepath):
        """Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def parse_label_line(self, label_file_line):
        # Parse data elements
        data = label_file_line.strip("\n").split(" ")
        if data[0] in ["avstack", "nuscenes"]:
            idx = 2
            ts = data[idx]
            idx += 1
            ID = data[idx]
            idx += 1
            obj_type = data[idx]
            idx += 1
            occ = Occlusion(int(data[idx]))
            idx += 1
            pos = data[idx : (idx + 3)]
            idx += 3
            t_box = pos
            vel = data[idx : (idx + 3)]
            idx += 3
            if np.all([v == "None" for v in vel]):
                vel = None
            acc = data[idx : (idx + 3)]
            idx += 3
            if np.all([a == "None" for a in acc]):
                acc = None
            box_size = data[idx : (idx + 3)]
            idx += 3
            q_O_to_obj = np.quaternion(*[float(d) for d in data[idx : (idx + 4)]])
            idx += 4
            if data[0] == "nuscenes":
                q_O_to_obj = q_O_to_obj.conjugate()
            ang = None
            where_is_t = data[idx]
            idx += 1
            object_reference = get_reference_from_line(" ".join(data[idx:]))
        elif data[0] == "kitti-v2":  # converted kitti raw data -- longitudinal dataset
            idx = 1
            ts = data[idx]
            idx += 1
            ID = data[idx]
            idx += 1
            obj_type = data[idx]
            idx += 1
            orientation = data[idx]
            idx += 1
            occ = Occlusion.UNKNOWN
            box2d = data[idx : (idx + 4)]
            idx += 4
            box_size = data[idx : (idx + 3)]
            idx += 3
            t_box = data[idx : (idx + 3)]
            idx += 3
            yaw = float(data[idx])
            idx += 1
            score = float(data[idx])
            idx += 1
            object_reference = get_reference_from_line(" ".join(data[idx:]))
            pos = t_box
            vel = acc = ang = None
            where_is_t = "bottom"
            q_V_to_obj = tforms.transform_orientation([0, 0, yaw], "euler", "quat")
            q_O_to_V = object_reference.q.conjugate()
            q_O_to_obj = q_V_to_obj * q_O_to_V
        elif label_file_line[0] == "{":
            obj = json.loads(label_file_line, cls=ObjectStateDecoder)
            return obj
        else:  # data[0] == "kitti":  # assume kitti with no prefix -- this is for kitti static dataset
            ts = 0.0
            ID = np.random.randint(low=0, high=1e6)  # not ideal but ensures unique IDs
            obj_type = data[0]
            occ = Occlusion.UNKNOWN
            box2d = data[4:8]
            t_box = data[11:14]  # x_C_2_Obj == x_O_2_Obj for O as camera
            where_is_t = "bottom"
            box_size = data[8:11]
            pos = t_box
            vel = acc = ang = None
            try:
                yaw = -float(data[14]) - np.pi / 2
            except ValueError:
                import pdb

                pdb.set_trace()
            object_reference = self.get_calibration(self.frames[0], "image-2").reference
            q_Ccam_to_Cstan = q_cam_to_stan
            q_Cstan_to_obj = tforms.transform_orientation([0, 0, yaw], "euler", "quat")
            q_O_to_obj = q_Cstan_to_obj * q_Ccam_to_Cstan
        try:
            ID = int(ID)
        except ValueError as e:
            pass
        pos = Position(np.array([float(p) for p in pos]), object_reference)
        vel = (
            Velocity(np.array([float(v) for v in vel]), object_reference)
            if vel is not None
            else None
        )
        acc = (
            Acceleration(np.array([float(a) for a in acc]), object_reference)
            if acc is not None
            else None
        )
        rot = Attitude(q_O_to_obj, object_reference)
        ang = (
            AngularVelocity(np.quaternion([float(a) for a in ang]), object_reference)
            if ang is not None
            else None
        )
        hwl = [float(b) for b in box_size]
        box3d = Box3D(pos, rot, hwl, obj_type=obj_type, where_is_t=where_is_t, ID=ID)
        obj = VehicleState(obj_type, ID)
        obj.set(
            t=float(ts),
            position=pos,
            box=box3d,
            velocity=vel,
            acceleration=acc,
            attitude=rot,
            angular_velocity=ang,
            occlusion=occ,
        )
        return obj


general_to_detection_class = {
    "animal": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.wheelchair": "ignore",
    "movable_object.debris": "ignore",
    "movable_object.pushable_pullable": "ignore",
    "static_object.bicycle_rack": "ignore",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}
_nominal_whitelist_types = [
    "car",
    "pedestrian",
    "bicycle",
    "truck",
    "bus",
    "motorcycle",
]
_nominal_ignore_types = []

occlusion_mapping = {
    1: Occlusion.MOST,
    2: Occlusion.PARTIAL,
    3: Occlusion.PARTIAL,
    4: Occlusion.NONE,
}


class _nuManager(BaseSceneManager):
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(self, nuX, nuX_can, data_dir, split="v1.0-mini", verbose=False):
        self.nuX = nuX
        self.nuX_can = nuX_can
        self.data_dir = data_dir
        self.split = split
        self.split_path = os.path.join(data_dir, split)


class _nuBaseDataset(BaseSceneDataset):
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(
        self,
        nuX,
        nuX_can,
        data_dir,
        split,
        verbose=False,
        whitelist_types=_nominal_whitelist_types,
        ignore_types=nominal_ignore_types,
    ):
        super().__init__(whitelist_types, ignore_types)
        self.nuX = nuX
        self.nuX_can = nuX_can
        if nuX_can is not None:
            self.vehicle_pose = self.nuX_can.get_messages(self.scene["name"], "pose")
            self.vehicle_pose_utime = np.array(
                [vp["utime"] for vp in self.vehicle_pose]
            )
        else:
            self.vehicle_pose = None
        self.data_dir = data_dir
        self.split = split
        self.split_path = os.path.join(data_dir, split)
        self.make_sample_records()
        self.w, self.l, self.h = [1.73, 4.084, 1.562]
        self.hwl = [self.h, self.w, self.l]
        self.token_ID_map = {}

    @property
    def frames(self):
        return list(self.sample_records.keys())

    def make_sample_records(self):
        raise NotImplementedError

    def _get_sample_metadata(self, sample_token):
        return self.nuX.get("sample", sample_token)

    def _get_sensor_record(self, frame, sensor=None):
        try:
            sr = self.nuX.get(
                "sample_data", self.sample_records[frame]["data"][sensor.upper()]
            )
        except KeyError:
            sr = self.nuX.get(
                "sample_data", self.sample_records[frame]["key_camera_token"]
            )
        return sr

    def _get_calib_data(self, frame, sensor):
        try:
            calib_data = self.nuX.get(
                "calibrated_sensor",
                self._get_sensor_record(frame, sensor)["calibrated_sensor_token"],
            )
        except KeyError as e:
            try:
                calib_data = self.nuX.get(
                    "calibrated_sensor",
                    self._get_sensor_record(frame, self.sensors[sensor])[
                        "calibrated_sensor_token"
                    ],
                )
            except KeyError as e2:
                raise e  # raise the first exception
        return calib_data

    def _get_sensor_file_name(self, frame, sensor=None):
        return os.path.join(
            self.data_dir,
            self._get_sensor_record(frame, self.get_sensor_name(sensor))["filename"],
        )

    def _load_frames(self, sensor: str = None):
        return self.frames

    def _load_calibration(self, frame, ego_reference, sensor=None):
        """
        W := global "world" frame
        E := ego frame
        S := sensor frame

        Reference frame has standard coordinates:
            x: forward
            y: left
            z: up

        ego reference frame is: "the center of the rear axle projected to the ground."
        """
        calib_data = self._get_calib_data(frame, sensor)
        x_E_2_S_in_E = np.array(calib_data["translation"])
        q_E_to_S = np.quaternion(*calib_data["rotation"]).conjugate()
        reference = ReferenceFrame(x=x_E_2_S_in_E, q=q_E_to_S, reference=ego_reference)
        sensor = sensor if sensor is not None else self.sensor_name(frame)
        if "CAM" in sensor:
            P = np.hstack((np.array(calib_data["camera_intrinsic"]), np.zeros((3, 1))))
            calib = calibration.CameraCalibration(
                reference, P, self.img_shape, channel_order="bgr"
            )
        else:
            calib = calibration.Calibration(reference)
        return calib

    def _load_timestamp(self, frame, sensor, utime=False):
        ut = self._get_sensor_record(frame, sensor)["timestamp"]
        if utime:
            return ut
        else:
            return ut / 1e6 - self.t0

    def _load_image(self, frame, sensor=None):
        img_fname = self._get_sensor_file_name(frame, sensor)
        return imread(img_fname)

    def _load_ego(self, frame):
        ref = GlobalOrigin3D
        if self.vehicle_pose is not None:
            try:
                frame_vp = np.argmin(
                    abs(self.vehicle_pose_utime - self.get_timestamp(frame, utime=True))
                )
                vp = self.vehicle_pose[frame_vp]
                t = vp["utime"] / 1e6 - self.t0
                x_G_to_E_in_G = Position(vp["pos"], ref)
                q_G_to_E = Attitude(np.quaternion(*vp["orientation"]).conjugate(), ref)
                q_E_to_G = q_G_to_E.q.conjugate()
                v_in_G = Velocity(q_mult_vec(q_E_to_G, np.array(vp["vel"])), ref)
                acc_in_G = Acceleration(
                    q_mult_vec(q_E_to_G, np.array(vp["accel"])), ref
                )
                ang_in_G = AngularVelocity(
                    np.quaternion(*q_mult_vec(q_E_to_G, np.array(vp["rotation_rate"]))),
                    ref,
                )
            except Exception as e:
                raise e  # need to handle this
        else:
            sd_record = self._get_sensor_record(frame, "LIDAR_TOP")
            ego = self.nuX.get("ego_pose", sd_record["ego_pose_token"])
            t = ego["timestamp"] / 1e6 - self.t0
            x_G_to_E_in_G = Position(np.array(ego["translation"]), ref)
            q_G_to_E = Attitude(np.quaternion(*ego["rotation"]).conjugate(), ref)
            q_E_to_G = q_G_to_E.conjugate()
            v_in_G = Velocity(ego["speed"] * q_E_to_G.forward_vector, ref)
            acc_in_G = Acceleration(
                q_mult_vec(q_E_to_G.q, np.array(ego["acceleration"])), ref
            )
            ang_in_G = AngularVelocity(
                np.quaternion(*q_mult_vec(q_E_to_G.q, np.array(ego["rotation_rate"]))),
                ref,
            )
        box3d = Box3D(x_G_to_E_in_G, q_G_to_E, self.hwl, ID=-1)

        # -- set up ego in global reference frame
        veh = VehicleState(obj_type="car")
        veh.set(
            t=t,
            position=x_G_to_E_in_G,
            velocity=v_in_G,
            box=box3d,
            attitude=q_G_to_E,
            acceleration=acc_in_G,
            angular_velocity=ang_in_G,
            occlusion=Occlusion.NONE,
        )
        return veh

    def _load_objects(
        self,
        frame,
        sensor=None,
        whitelist_types=["car", "pedestrian", "bicycle", "truck", "bus", "motorcycle"],
        ignore_types=[],
    ):
        """
        automatically loads into local sensor coordinates

        class_names = [
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

        NOTE: for now, velocity is in the sensor's coordinate frame (moving)
        """
        sensor_data = self._get_sensor_record(frame, sensor)
        reference = self.get_calibration(frame, sensor).reference

        # get data for nuscenes
        objects = []
        _, boxes, _ = self.nuX.get_sample_data(sensor_data["token"])
        for box in boxes:
            metadata = self.nuX.get("sample_annotation", box.token)
            obj_type = general_to_detection_class[box.name]
            if (
                (obj_type in whitelist_types)
                or (whitelist_types == "all")
                or ("all" in whitelist_types)
            ):
                obj_token = metadata[
                    "instance_token"
                ]  # consistent between frames of a scene
                if obj_token not in self.token_ID_map:
                    self.token_ID_map[obj_token] = len(self.token_ID_map)
                obj_ID = self.token_ID_map[obj_token]
                position = Position(box.center, reference)
                velocity = Velocity(self.nuX.box_velocity(box.token), GlobalOrigin3D)
                velocity.change_reference(reference, inplace=True)
                acceleration = None
                attitude = Attitude(
                    np.quaternion(*box.orientation).conjugate(), reference
                )
                angular_velocity = None
                hwl = [box.wlh[2], box.wlh[0], box.wlh[1]]
                box = Box3D(position, attitude, hwl, where_is_t="center", ID=obj_ID)
                occlusion = occlusion_mapping[int(metadata["visibility_token"])]
                obj = VehicleState(obj_type=obj_type, ID=obj_ID)
                obj.set(
                    t=sensor_data["timestamp"] / 1e6 - self.t0,
                    position=position,
                    velocity=velocity,
                    acceleration=acceleration,
                    attitude=attitude,
                    angular_velocity=angular_velocity,
                    box=box,
                    occlusion=occlusion,
                )
                obj.change_reference(reference, inplace=True)
                objects.append(obj)
        return np.array(objects)

    def _number_objects_from_file(self, frame, **kwargs):
        sensor_data = self._get_sensor_record(frame, "LIDAR_TOP")
        _, boxes, _ = self.nuX.get_sample_data(sensor_data["token"])
        return len(boxes)

    # def _get_anns_metadata(self, sample_data_token):
    #     try:
    #         _, boxes, camera_intrinsic = self.nuX.get_sample_data(sample_data_token)
    #         boxes = [{"name": box.name, "box": box} for box in boxes]
    #     except AttributeError:
    #         object_tokens, surface_tokens = self.nuX.list_anns(
    #             sample_data_token, verbose=False
    #         )
    #         obj_anns = [
    #             self.nuX.get("object_ann", obj_tok) for obj_tok in object_tokens
    #         ]
    #         boxes = [
    #             {
    #                 "name": self.nuX.get("category", obj_ann["category_token"])["name"],
    #                 "box": obj_ann["bbox"],
    #             }
    #             for obj_ann in obj_anns
    #         ]
    #     velocities = [self.nuX.box_velocity(box["box"].token) for box in boxes]
    #     return boxes, velocities
    # try:
    #     boxes, velocities = self._get_anns_metadata)
    # except KeyError:
    #     boxes, velocities = self._get_anns_metadata(sensor_data["sample_token"])
    # objects = []
    # for box, vel in zip(boxes, velocities):
    #     obj_type = general_to_detection_class[box["name"]]
    #     if (
    #         (obj_type in whitelist_types)
    #         or (whitelist_types == "all")
    #         or ("all" in whitelist_types)
    #     ):
    #         ts = sensor_data["timestamp"]
    #         import pdb; pdb.set_trace()
    #         line = self._box_to_line(ts, box["box"], reference)
    #         obj = self.parse_label_line(line)
    #         velocity = Velocity(vel, reference=ego.reference)
    #         velocity.change_reference(reference, inplace=True)
    #         obj.velocity = velocity
    #         objects.append(obj)
    # return np.array(objects)

    def _box_to_line(self, ts, box, reference):
        ID = box.token
        obj_type = general_to_detection_class[box.name]
        vel = [None] * 3
        acc = [None] * 3
        w, l, h = box.wlh
        pos = box.center
        q = box.orientation
        occ = Occlusion.UNKNOWN
        line = (
            f"nuscenes object_3d {ts} {ID} {obj_type} {int(occ)} {pos[0]} "
            f"{pos[1]} {pos[2]} {vel[0]} {vel[1]} {vel[2]} {acc[0]} "
            f'{acc[1]} {acc[2]} {h} {w} {l} {q[0]} {q[1]} {q[2]} {q[3]} {"center"} {reference.format_as_string()}'
        )
        return line

    def _ego_to_line(self, ts, ego):
        obj_type = "car"
        ID = ego["token"]
        pos = np.array(ego["translation"])
        q = np.quaternion(*ego["rotation"]).conjugate()
        if self.ego_speed_interp is not None:
            vel = tforms.transform_orientation(q, "quat", "dcm")[
                :, 0
            ] * self.ego_speed_interp(ts)
        else:
            vel = [None] * 3
        acc = [None] * 3
        w, l, h = [1.73, 4.084, 1.562]
        reference = GlobalOrigin3D
        occ = Occlusion.NONE
        line = (
            f"nuscenes object_3d {ts} {ID} {obj_type} {int(occ)} {pos[0]} "
            f"{pos[1]} {pos[2]} {vel[0]} {vel[1]} {vel[2]} {acc[0]} "
            f'{acc[1]} {acc[2]} {h} {w} {l} {q.w} {q.x} {q.y} {q.z} {"center"} {reference.format_as_string()}'
        )
        return line
