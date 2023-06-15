# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-29
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-10-21
# @Description:
"""

"""

import math

import carla
import numpy as np
from avstack.geometry import transformations as tforms
from avstack.geometry import NominalOriginStandard, Rotation, Translation, VectorDirMag, bbox
from avstack.environment.objects import VehicleState


def get_obj_type_from_actor(actor):
    h = 2 * actor.bounding_box.extent.z
    w = 2 * actor.bounding_box.extent.y
    l = 2 * actor.bounding_box.extent.x
    if "vehicle" in actor.type_id:
        n_wheels = actor.attributes["number_of_wheels"]
        if int(n_wheels) == 4:
            if h >= 2:
                obj_type = "truck"
            else:
                obj_type = "car"
        elif int(n_wheels) == 2:
            if any([mot in actor.type_id for mot in ["harley", "kawasaki", "yamaha"]]):
                obj_type = "motorcycle"
            else:
                obj_type = "bicycle"
        else:
            raise NotImplementedError(n_wheels)
    elif "walker" in actor.type_id:
        obj_type = "pedestrian"
        raise NotImplementedError(obj_type)
    else:
        raise NotImplementedError(actors.type_id)
    return obj_type


def wrap_actor_to_vehicle_state(t, actor):
    obj_type = get_obj_type_from_actor(actor)
    h = 2 * actor.bounding_box.extent.z
    w = 2 * actor.bounding_box.extent.y
    l = 2 * actor.bounding_box.extent.x
    VS = VehicleState(obj_type, actor.id)
    tf = actor.get_transform()
    v = actor.get_velocity()
    ac = actor.get_acceleration()
    pos = Translation(
        [tf.location.x, -tf.location.y, tf.location.z], NominalOriginStandard
    )
    vel = VectorDirMag([v.x, -v.y, v.z], NominalOriginStandard)
    acc = VectorDirMag([ac.x, -ac.y, ac.z], NominalOriginStandard)
    q_OR_to_obj = tforms.transform_orientation(
        carla_rotation_to_RPY(tf.rotation), "euler", "quat"
    )
    # q_OR_to_obj = tforms.transform_orientation(carla_rotation_to_RPY(tf.rotation), 'euler', 'quat').conjugate()

    av = actor.get_angular_velocity()
    ang = VectorDirMag([av.x, -av.y, av.z], NominalOriginStandard)
    box = bbox.Box3D(
        [h, w, l, pos.vector, q_OR_to_obj], NominalOriginStandard, where_is_t="bottom"
    )
    att = Rotation(box.q, NominalOriginStandard)
    VS.set(t, pos, box, vel, acc, att, ang)
    return VS


def carla_rotation_to_RPY(carla_rotation):
    """
    Convert a carla rotation to a roll, pitch, yaw tuple
    Considers the conversion from left-handed system (unreal) to right-handed
    system (ROS).
    Considers the conversion from degrees (carla) to radians (ROS).
    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a tuple with 3 elements (roll, pitch, yaw)
    :rtype: tuple
    """
    roll = math.radians(carla_rotation.roll)
    pitch = -math.radians(carla_rotation.pitch)
    yaw = -math.radians(carla_rotation.yaw)

    return [roll, pitch, yaw]


def carla_rotation_to_quaternion(carla_rotation):
    rpy = carla_rotation_to_RPY(carla_rotation)
    return tforms.transform_orientation(rpy, "euler", "quat")


def carla_location_to_numpy_vector(carla_location):
    """
    Convert a carla location to a ROS vector3
    Considers the conversion from left-handed system (unreal) to right-handed
    system (ROS)
    :param carla_location: the carla location
    :type carla_location: carla.Location
    :return: a numpy.array with 3 elements
    :rtype: numpy.array
    """
    return np.array([carla_location.x, -carla_location.y, carla_location.z])


def numpy_vector_to_carla_location(x):
    return carla.Location(x[0], -x[1], x[2])


def quaternion_to_carla_rotation(q):
    rpy = tforms.transform_orientation(q, "quat", "euler")
    return carla.Rotation(
        pitch=-180 / np.pi * rpy[1],
        yaw=-180 / np.pi * rpy[2],
        roll=180 / np.pi * rpy[0],
    )
