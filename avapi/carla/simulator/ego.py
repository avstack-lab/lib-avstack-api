# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2021-10-26
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-10-22
# @Description:
"""
Define the ego classes
"""


import random

import numpy as np
from avstack import GroundTruthInformation
from avstack.datastructs import DataManager
from avstack.environment import EnvironmentState
from avstack.geometry import (
    Acceleration,
    AngularVelocity,
    Attitude,
    GlobalOrigin3D,
    Pose,
    Position,
    ReferenceFrame,
    Velocity,
)
from avstack.geometry import transformations as tforms
from avstack.modules.perception import detections
from carla import Location, VehicleControl
from pygame.locals import K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_UP, K_q

from avapi.carla.simulator import sensors, utils


class CarlaEgoActor:
    def __init__(self, world, ego_stack, cfg):
        # spawn actor
        self.world = world
        self.debug = world.debug
        self.map = world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.vehicle_bps = self.world.get_blueprint_library().filter("vehicle")
        self.cfg = cfg
        self._ego_stack_template = ego_stack
        self._spawn_actor()

        self.t0 = None
        self.frame0 = None
        e_pose = self.get_ego_pose()
        self.reference = ReferenceFrame(
            e_pose.position.x, e_pose.attitude.q, GlobalOrigin3D
        )

    def _spawn_actor(self):
        # -- vehicle blueprint
        if self.cfg["idx_vehicle"] in ["random", "randint"]:
            bp = np.random.choice(self.vehicle_bps)
        elif isinstance(self.cfg["idx_vehicle"], int):
            bp = self.vehicle_bps[self.cfg["idx_vehicle"]]
        elif (
            isinstance(self.cfg["idx_vehicle"], str)
            and len(self.cfg["idx_vehicle"]) > 1
        ):
            bp = self.vehicle_bps.filter(self.cfg["idx_vehicle"])[0]
        else:
            raise NotImplementedError

        # -- spawn point
        if self.cfg["idx_spawn"] in ["random", "randint"]:
            tf = np.random.choice(self.spawn_points)
        elif isinstance(self.cfg["idx_spawn"], int):
            tf = self.spawn_points[self.cfg["idx_spawn"]]
        else:
            raise NotImplementedError(type(self.cfg["idx_spawn"]))

        # -- spawn actor
        n_att = 10
        d_inc = 3
        i = 0
        while i < n_att:
            self.actor = self.world.try_spawn_actor(bp, tf)
            if self.actor is None:
                loc = np.array([tf.location.x, tf.location.y, tf.location.z])
                fv = tf.rotation.get_forward_vector()
                tf.location.x += d_inc * fv.x
                tf.location.y += d_inc * fv.y
                # tf.location.z += d_inc * fv.z
                i += 1
            else:
                break
        else:
            raise RuntimeError(f"Could not spawn ego actor after {i} attempts")

        self.sensors = {}
        self.sensor_IDs = {}
        self.sensor_data_manager = DataManager()
        self._spd_temp = None
        self._i_spd_temp = 0
        self.world.tick()  # to allow for initialization
        try:
            # provide initialization
            t_init = 0
            ego_init = self.get_vehicle_data_from_actor(t_init)

            # initialize algorithms
            self.algorithms = self._ego_stack_template(
                t_init, ego_init, map_data=self.map
            )
            if self.cfg["idx_destination"] is not None:
                if self.cfg["idx_destination"] == "random":
                    dest = random.choice(self.spawn_points)
                else:
                    dest = self.spawn_points[self.cfg["idx_destination"]].location
                    dest = [dest.x, -dest.y, dest.z]
            elif self.cfg["delta_destination"] is not None:
                dkeys = sorted(list(self.cfg["delta_destination"].keys()))
                if dkeys == ["forward", "right", "up"]:
                    delta = np.array([self.cfg["delta_destination"][k] for k in dkeys])
                    dest = ego_init.position.vector
                    dest = ego_init.position.vector + ego_init.attitude.T @ delta
                elif dkeys == ["x", "y", "z"]:
                    raise
                else:
                    raise NotImplementedError(dkeys)
            else:
                dest = None
            if dest is not None:
                dest_true = self.algorithms.set_destination(dest, coordinates="avstack")
            else:
                dest_true = None
            self.destination = dest_true
            self.roaming = self.cfg["roaming"]
        except (KeyboardInterrupt, Exception) as e:
            self.destroy()
            raise e

        # -- start autopilot, if enabled
        if self.cfg["autopilot"]:
            print("Enabling ego autopilot")
            if not self.algorithms.is_passthrough:
                raise RuntimeError(
                    "Cannot set autopilot unless algorithms are passthrough"
                )
            self.actor.set_autopilot(True)

    def initialize(self, t0, frame0):
        self.t0 = t0
        self.frame0 = frame0
        for k1, sens in self.sensors.items():
            sens.initialize(t0, frame0)

    def restart(self, t0, frame0, save_folder):
        from .bootstrap import bootstrap_ego_sensor

        self.destroy()
        self._spawn_actor()
        # --- make sensors attached to ego
        try:
            # # TODO: MOVE THE SENSOR OPTIONS TO A HIGHER LEVEL SOMEWHERE
            # sensor_options = {
            #     "camera": sensors.RgbCameraSensor,
            #     "gnss": sensors.GnssSensor,
            #     "gps": sensors.GnssSensor,
            #     "depthcam": sensors.DepthCameraSensor,
            #     "imu": sensors.ImuSensor,
            #     "lidar": sensors.LidarSensor,
            # }
            # for sens in sensor_options.values():
            #     sens.reset_next_id()
            for i, cfg_sensor in enumerate(self.cfg["sensors"]):
                bootstrap_ego_sensor(self, i, cfg_sensor, save_folder)
        except (KeyboardInterrupt, Exception) as e:
            self.destroy()
            raise e
        self.initialize(t0, frame0)

    def get_ego_pose(self):
        tf = self.actor.get_transform()
        q = tforms.transform_orientation(
            utils.carla_rotation_to_RPY(tf.rotation), "euler", "quat"
        )
        # center of the vehicle
        pos = Position([tf.location.x, -tf.location.y, tf.location.z], GlobalOrigin3D)
        att = Attitude(q, GlobalOrigin3D)
        return Pose(pos, att)

    def get_vehicle_data_from_actor(self, t):
        try:
            return utils.wrap_actor_to_vehicle_state(t, self.actor)
        except RuntimeError as e:
            return None

    def get_object_data_from_world(self, t):
        objects = []
        for act in self.world.get_actors():
            if "vehicle" in act.type_id:
                obj_type = "Vehicle"
            elif "walker" in act.type_id:
                obj_type = "Pedestrian"
            elif (
                (act.type_id in ["spectator"])
                or ("traffic" in act.type_id)
                or ("sensor" in act.type_id)
            ):
                continue
            else:
                raise NotImplementedError(f"{act.type_id}, {act}")
            if act.get_location().distance(self.actor.get_location()) > 1 / 2:
                obj_data = utils.wrap_actor_to_vehicle_state(t, act)
                if obj_data is not None:
                    objects.append(obj_data)
        return objects

    def get_lane_lines(self, debug=False):
        """Gets lane lines in local coordinates of ego"""
        pose_g2l = self.get_ego_pose()
        wpt_init = self.map.get_waypoint(
            self.actor.get_location(), project_to_road=True
        )
        wpts = wpt_init.next_until_lane_end(distance=1)
        if (wpts is None) or (len(wpts) < 3):
            # lanes = [None, None]
            lanes = []
        else:
            wpts_local = [
                [
                    Position(
                        utils.carla_location_to_numpy_vector(wpt.transform.location),
                        GlobalOrigin3D,
                    ).change_reference(self.reference, inplace=False),
                    wpt.lane_width,
                ]
                for wpt in wpts
            ]
            pts_left = [
                Position([wpt[0], wpt[1] + lane_width / 2, wpt[2]], GlobalOrigin3D)
                for wpt, lane_width in wpts_local
            ]
            pts_right = [
                Position([wpt[0], wpt[1] - lane_width / 2, wpt[2]], GlobalOrigin3D)
                for wpt, lane_width in wpts_local
            ]
            lane_left = detections.LaneLineInSpace(pts_left)
            lane_right = detections.LaneLineInSpace(pts_right)
            if debug:
                T_l2g = pose_g2l.matrix
                for i in range(len(wpts) - 1):
                    # draw center
                    self.debug.draw_line(
                        wpts[i].transform.location,
                        wpts[i + 1].transform.location,
                        life_time=0.5,
                    )
                    # draw left and right
                    p1l = T_l2g @ pts_left[i]
                    p2l = T_l2g @ pts_left[i + 1]
                    self.debug.draw_line(
                        Location(p1l.x, -p1l.y, p1l.z),
                        Location(p2l.x, -p2l.y, p2l.z),
                        life_time=0.5,
                    )
                    p1r = T_l2g @ pts_right[i]
                    p2r = T_l2g @ pts_right[i + 1]
                    self.debug.draw_line(
                        Location(p1r.x, -p1r.y, p1r.z),
                        Location(p2r.x, -p2r.y, p2r.z),
                        life_time=0.5,
                    )
            lanes = [lane_left, lane_right]
        return lanes

    def get_ground_truth(self, t_elapsed, frame, speed_limit=8):
        environment = EnvironmentState()
        environment.speed_limit = speed_limit
        if self._spd_temp is not None:
            if self._i_spd_temp < self._n_spd_temp:
                print(f"setting speed limit to {self._spd_temp} for now")
                environment.speed_limit = self._spd_temp
                self._i_spd_temp += 1
            else:
                self._spd_temp = None
                self._i_spd_temp = 0
        ego_state = self.get_vehicle_data_from_actor(t_elapsed)
        assert ego_state is not None
        objects = self.get_object_data_from_world(t_elapsed)
        lane_lines = self.get_lane_lines()
        return GroundTruthInformation(
            frame=frame,
            timestamp=t_elapsed,
            ego_state=ego_state,
            objects=objects,
            lane_lines=lane_lines,
            environment=environment,
        )

    def tick(self, t_elapsed, frame, infrastructure=None):
        # -- update ground truth
        ground_truth = self.get_ground_truth(t_elapsed, frame)
        self.reference.x = ground_truth.ego_state.position.x
        self.reference.q = ground_truth.ego_state.attitude.q

        # -- apply algorithms
        ctrl, alg_debug = self.algorithms.tick(
            frame,
            t_elapsed,
            self.sensor_data_manager,
            infrastructure=infrastructure,
            ground_truth=ground_truth,
        )
        if ctrl is not None:
            self.apply_control(ctrl)

        # -- check if we need to set new destination
        done = False
        if self.destination is not None:
            d_dest = ground_truth.ego_state.position.distance(self.destination)
            if (self.destination is not None) and d_dest < 20:
                if self.roaming:
                    dest = self.random_spawn().location
                    dest = np.array(
                        [dest.x, -dest.y, dest.z]
                    )  # put into avstack coordinates
                    dest_true = self.algorithms.set_destination(
                        dest, coordinates="avstack"
                    )
                    self.destination = dest_true
                else:
                    done = True
        debug = {
            "algorithms": alg_debug,
        }
        return done, debug

    def random_spawn(self):
        return np.random.choice(self.spawn_points)

    def draw_waypoint(self, plan):
        wpt = plan.top()[1]
        loc = Location(wpt.location.x, -wpt.location.y, wpt.location.z)
        self.debug.draw_point(loc, size=1 / 2, life_time=1 / 2)

    def destroy(self):
        if self.actor is not None:
            try:
                self.actor.destroy()
            except RuntimeError as e:
                pass  # usually because already destroyed
        for s_name, sensor in self.sensors.items():
            try:
                sensor.destroy()
            except (KeyboardInterrupt, Exception) as e:
                print(f"Could not destroy sensor {s_name}...continuing")

    def apply_control(self, ctrl):
        VC = VehicleControl(
            ctrl.throttle, ctrl.steer, ctrl.brake, ctrl.hand_brake, ctrl.reverse
        )
        self.actor.apply_control(VC)

    def add_sensor(self, sensor_name, sensor):
        assert sensor_name not in self.sensors
        self.sensors[sensor_name] = sensor
        self.sensor_IDs[sensor_name] = sensor.ID
        print(f"Added {sensor_name} sensor")

    def set_control_mode(self, mode):
        assert mode in ["autopilot", "manual"]
        if self.control_mode != mode:
            print(f"Setting control to: {mode} mode")
        else:
            print(f"Control already in {mode} mode")
        self.control_mode = mode

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP]:
            throttle = min(self.last_control.throttle + 0.01, 1)
        else:
            throttle = 0.0
        if keys[K_DOWN]:
            brake = min(self.last_control.brake + 0.2, 1)
        else:
            brake = 0
        steer_increment = 5e-4 * milliseconds  # to adjust sensitivity of steering
        if keys[K_LEFT]:
            if self.last_control.steer > 0:
                steer = 0
            else:
                steer = self.last_control.steer - steer_increment
        elif keys[K_RIGHT]:
            if self.last_control.steer < 0:
                steer = 0
            else:
                steer = self.last_control.steer + steer_increment
        else:
            steer = 0.0
        steer = round(min(0.7, max(-0.7, steer)), 2)
        hand_brake = keys[K_SPACE]
        reverse = keys[K_q]  # hold down q to go into reverse
        gear = 1 if reverse else -1
        return carla.VehicleControl(
            throttle, steer, brake, hand_brake, reverse, gear=gear
        )
