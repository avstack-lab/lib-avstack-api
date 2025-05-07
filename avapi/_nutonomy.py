import os

import numpy as np
from avstack import calibration
from avstack.environment.objects import Occlusion, VehicleState
from avstack.geometry import (
    Acceleration,
    AngularVelocity,
    Attitude,
    Box3D,
    GlobalOrigin3D,
    Position,
    ReferenceFrame,
    Velocity,
    q_mult_vec,
)
from avstack.geometry import transformations as tforms
from avstack.modules.control import VehicleControlSignal
from cv2 import imread

from ._dataset import BaseSceneDataset, BaseSceneManager


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
            # get pose information
            self.vehicle_pose = self.nuX_can.get_messages(self.scene, "pose")
            self.vehicle_pose_utime = np.array(
                [vp["utime"] for vp in self.vehicle_pose]
            )

            # get vehicle monitor data
            self.vehicle_monitor = self.nuX_can.get_messages(
                self.scene, "vehicle_monitor"
            )
            self.vehicle_monitor_utime = np.array(
                [vm["utime"] for vm in self.vehicle_monitor]
            )

            # get zoe sensor data
            self.vehicle_zoe = self.nuX_can.get_messages(self.scene, "zoesensors")
            self.vehicle_zoe_utime = np.array([vz["utime"] for vz in self.vehicle_zoe])
        else:
            self.vehicle_pose = None
        self.data_dir = data_dir
        self.split = split
        self.split_path = os.path.join(data_dir, split)
        self.make_sample_records()
        self.w, self.l, self.h = [1.73, 4.084, 1.562]
        self.hwl = [self.h, self.w, self.l]
        self.token_ID_map = {}
        self.verbose = verbose

    @property
    def frames(self):
        return list(self.sample_records.keys())

    @property
    def timestamps(self):
        return list(
            self.sample_records[frame]["timestamp"] / 1e6 - self.t0
            for frame in self.frames
        )

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

    def _get_sensor_file_name(self, frame, sensor=None, agent=None):
        return os.path.join(
            self.data_dir,
            self._get_sensor_record(frame, self.get_sensor_name(sensor, agent))[
                "filename"
            ],
        )

    def _load_frames(self, **kwargs):
        return self.frames

    def _load_timestamps(self, **kwargs):
        return self.timestamps

    def _load_agent_set(self, frame: int) -> set:
        # TODO: this is slow...improve
        return {ag.ID for ag in self.get_agents(frame)}

    def _load_calibration(self, frame, ego_reference, sensor=None, **kwargs):
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

    def _load_timestamp(self, frame, sensor, utime=False, **kwargs):
        if sensor is None:
            sensor = "LIDAR_TOP"
        ut = self._get_sensor_record(frame, sensor)["timestamp"]
        if utime:
            return ut
        else:
            return ut / 1e6 - self.t0

    def _load_image(self, frame, sensor=None, **kwargs):
        img_fname = self._get_sensor_file_name(frame, sensor)
        return imread(img_fname)

    def _load_control_signal(
        self, frame, bounded=False, **kwargs
    ) -> VehicleControlSignal:
        """Loads the vehicle control signal from CAN data

        vehicle monitor:
            steering: angle in degrees at a resolution of 0.1 in range [-780, 779.9]
            throttle: throttle pedal position as an integer in range [0, 1000].
            brake: braking pressure in bar. An integer in range [0, 126]
        """

        # load the signals
        timestamp = self.get_timestamp(frame, utime=True)
        frame_vm = np.argmin(abs(self.vehicle_monitor_utime - timestamp))
        vm = self.vehicle_monitor[frame_vm]

        # scale values (TODO: determine the best way to do this)
        t = vm["utime"] / 1e6 - self.t0
        steer = np.pi / 180 * vm["steering"]
        throttle = vm["throttle"] / 1000.0
        brake = vm["brake"] / 126.0

        # package into a data structure
        signal = VehicleControlSignal(
            timestamp=t,
            throttle=throttle,
            brake=brake,
            steer=steer,
            hand_brake=False,
            reverse=False,  # it never reverses
            manual_gear_shift=False,
            bounded=bounded,
        )

        return signal

    def _load_route(self, **kwargs):
        raise NotImplementedError

    def _load_ego(self, frame, **kwargs) -> VehicleState:
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
        box3d = Box3D(x_G_to_E_in_G, q_G_to_E, self.hwl, ID=-1, where_is_t="bottom")

        # -- set up ego in global reference frame
        veh = VehicleState(obj_type="car", ID=1000)
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
        **kwargs,
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
