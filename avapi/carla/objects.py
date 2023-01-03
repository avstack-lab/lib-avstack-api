# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-07-28
# @Description:
"""

"""

from .kitti import KittiObject3dLabel


CarlaObject3dLabel = KittiObject3dLabel


def get_carla_label_text_from_2d_box(obj_type, sensor_source, box_2d, score):
    return f'carla, 2d-box, {sensor_source}, {obj_type}, {box_2d.xmin}, ' + \
           f'{box_2d.ymin}, {box_2d.xmax}, {box_2d.ymax}, {score}'


def get_carla_label_text_from_3d_box(obj_type, sensor_source, box_3d, score):
    return f'carla, 3d-box, {sensor_source}, {obj_type}, {box_3d.t[0]}, {box_3d.t[1]}, ' + \
           f'{box_3d.t[2]}, {box_3d.h}, {box_3d.w}, {box_3d.l}, {box_3d.yaw}, {score}'
