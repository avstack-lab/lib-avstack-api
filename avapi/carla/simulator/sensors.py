# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2021-10-26
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-10-21
# @Description:
"""

"""
import abc
import itertools
import math
import os
import queue
import time
import weakref
from queue import PriorityQueue

import avstack.sensors
import carla
import numpy as np
import quaternion
from avstack import transformations as tforms
from avstack.calibration import Calibration, CameraCalibration
from avstack.geometry import (
    NominalOriginStandard,
    Origin,
    Rotation,
    Translation,
    q_stan_to_cam,
)
from carla import ColorConverter as cc

from avapi.carla import utils


SensorData = avstack.sensors.SensorData

# =============================================================
# SENSORS
# =============================================================


class Sensor:
    id_iter = itertools.count()
    blueprint_name = ""
    name = ""

    def __init__(
        self,
        source_name,
        parent,
        tform,
        attr,
        mode,
        noise,
        save=False,
        save_folder="./sensor_data",
    ):

        # -- attributes
        self.parent = parent
        self.tform_to_parent = tform
        self.global_ID = next(Sensor.id_iter)
        self.lla_origin = parent.map.transform_to_geolocation(carla.Location(0, 0, 0))
        self.mode = mode
        self.noise = noise
        self.t0 = None
        self.frame0 = None
        self.initialized = False

        # -- identifiers
        source_ID = next(self.next_id)
        self.ID = source_ID
        self.source_ID = source_ID
        self.source_name = source_name
        self.source_identifier = source_name + "-" + str(source_ID)

        # -- origin and calibration information
        loc = tform.location
        x = np.array([loc.x, -loc.y, loc.z])
        q_c = utils.carla_rotation_to_quaternion(tform.rotation)

        # -- check if its a camera
        try:
            # -- camera!
            w = int(attr["image_size_x"])
            h = int(attr["image_size_y"])
            fov_h = float(attr["fov"]) / 2.0  # half horizontal FOV
            f = (w / 2) / (np.tan(fov_h * math.pi / 180.0))
            self.P = np.array([[f, 0, w / 2.0, 0], [0, f, h / 2.0, 0], [0, 0, 1, 0]])

            # -- cameras have different default coordinates
            q_c = q_stan_to_cam * q_c
        except KeyError as e:
            # -- not a camera!
            self.P = None
        self.origin = Origin(x, q_c)
        if self.P is None:
            self.calibration = Calibration(self.origin)
        else:
            imsize = (h, w)
            self.calibration = CameraCalibration(self.origin, self.P, imsize)

        # -- spawn from blueprint
        self.attr = attr
        bp = parent.world.get_blueprint_library().find(self.blueprint_name)
        for k, v in attr.items():
            bp.set_attribute(k, str(v))
        self.object = parent.world.spawn_actor(
            bp,
            tform,
            attach_to=self.parent.actor,
            attachment_type=carla.AttachmentType.Rigid,
        )
        time.sleep(0.5)  # to allow for initialization

        # -- saving
        self.save = save
        self.save_folder = save_folder

        # -- callback
        weak_self = weakref.ref(self)
        self.object.listen(lambda event: self._on_sensor_event(weak_self, event))

    @property
    def _default_subfolder(self):
        return self.source_identifier

    @classmethod
    def reset_next_id(cls):
        cls.next_id = itertools.count()

    def initialize(self, t0, frame0):
        self.t0 = t0
        self.frame0 = frame0
        self.initialized = True

    def destroy(self):
        self.object.destroy()

    @abc.abstractmethod
    def _on_sensor_event(weak_self):
        """implemented in subclasses"""

    def _make_data_class(self, timestamp, frame, data, **kwargs):
        if self.initialized:
            data_class = self.base_data(
                timestamp=timestamp - self.t0,
                frame=frame - self.frame0,
                data=data,
                calibration=self.calibration,
                source_ID=self.source_ID,
                source_name=self.source_name,
                **kwargs
            )
            self.parent.sensor_data_manager.push(data_class)
            if self.save:
                data_class.save_to_folder(
                    self.save_folder, add_subfolder=True, **kwargs
                )
        else:
            print("sensor not initialized")


class GnssSensor(Sensor):
    next_id = itertools.count()
    blueprint_name = "sensor.other.gnss"
    name = "gps"
    base_data = avstack.sensors.GpsData

    @staticmethod
    def _on_sensor_event(weak_self, gnss):
        self = weak_self()
        lla = [np.pi / 180 * gnss.latitude, np.pi / 180 * gnss.longitude, gnss.altitude]
        ned = tforms.transform_point(lla, "lla", "ned", (np.array([0, 0, 0]), "lla"))
        sR = self.noise["sigma"]
        sb = self.noise["bias"]
        b = np.array([sb["east"], sb["north"], sR["up"]])
        r = np.array([sR["east"], sR["north"], sR["up"]])
        R = np.diag(r**2)
        v_enu = np.squeeze(np.array([ned[1], ned[0], -ned[2]]))
        v_enu = v_enu + b + r * np.random.randn(3)
        enu = {"z": v_enu, "R": R}
        self._make_data_class(gnss.timestamp, gnss.frame, enu, levar=self.origin.x)


class ImuSensor(Sensor):
    next_id = itertools.count()
    blueprint_name = "sensor.other.imu"
    name = "imu"
    base_data = avstack.sensors.ImuData

    @staticmethod
    def _on_sensor_event(weak_self, imu):
        self = weak_self()
        agc = {
            "accelerometer": imu.accelerometer,
            "gyroscope": imu.gyroscope,
            "compass": imu.compass,
        }
        self._make_data_class(imu.timestamp, imu.frame, agc)


class RgbCameraSensor(Sensor):
    next_id = itertools.count()
    blueprint_name = "sensor.camera.rgb"
    name = "camera"
    base_data = avstack.sensors.ImageData

    @staticmethod
    def _on_sensor_event(weak_self, image):
        self = weak_self()
        image.convert(cc.Raw)
        self._make_data_class(image.timestamp, image.frame, image)


class DepthCameraSensor(Sensor):
    next_id = itertools.count()
    blueprint_name = "sensor.camera.depth"
    name = "camera"
    base_data = avstack.sensors.DepthImageData

    @staticmethod
    def _on_sensor_event(weak_self, image):
        self = weak_self()
        self._make_data_class(image.timestamp, image.frame, image)


class LidarSensor(Sensor):
    next_id = itertools.count()
    blueprint_name = "sensor.lidar.ray_cast"
    name = "lidar"
    base_data = avstack.sensors.LidarData

    @staticmethod
    def _on_sensor_event(weak_self, pc):
        self = weak_self()
        self._make_data_class(pc.timestamp, pc.frame, pc, flipy=True)


class RadarSensor(Sensor):
    next_id = itertools.count()
    blueprint_name = "sensor.other.radar"
    name = "radar"
    base_data = avstack.sensors.RadarDataRazelRRT

    @staticmethod
    def _on_sensor_event(weak_self, razelrrt):
        self = weak_self()
        self._make_data_class(razelrrt.timestamp, razelrrt.frame, razelrrt)
