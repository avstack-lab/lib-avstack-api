# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-03
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-11
# @Description:
"""

"""

import os
from avstack.objects import VehicleState


def get_timestamps(src_folder):
    ts = []
    with open(os.path.join(src_folder, 'timestamps.txt')) as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line)==0: continue
            ts.append(line)
    return ts


def check_xor_for_none(a, b):
    assert check_xor(a is None, b is None), "Can only pass in one of these inputs"


def check_xor(a, b):
    return (a or b) and (not a or not b)


def remove_glob(glob_files):
    rem = False
    for f in glob_files:
        os.remove(f)
        rem = True
    if rem:
        print('Removed files from: {}'.format(os.path.dirname(f)), flush=True)


def get_indices_in_folder(glob_dir, idxs=None):
    """Get indices of items in a glob_dir
    optionally: enforce that they are in the list idxs
    """
    idxs_available = []
    for f in glob_dir:
        if 'log' in f:
            continue
        idx = int(f.split('/')[-1].replace('.txt',''))
        if idxs is not None:
            try:
                iterator = iter(idxs)
            except TypeError:
                # not iterable
                if not (idx == idxs):
                    continue
            else:
                # iterable
                if idx not in idxs:
                    continue
        if idx not in idxs_available:
            idxs_available.append(idx)
    return idxs_available


# def get_tracks_from_label_file(label_file, whitelist_types='all', ignore_types=[]):
#     assert os.path.exists(label_file), f'Cannot find requested label file at {label_file}'
#     with open(label_file, 'r') as f:
#         lines = [line.rstrip() for line in f]
#     tracks = [get_track_from_label_text(line) for line in lines]
#     if (whitelist_types != 'all') and ('all' not in whitelist_types):
#         tracks = [trk for trk in tracks if (trk.obj_type not in ignore_types) & (trk.obj_type in whitelist_types)]
#     return np.asarray(tracks)


# def get_object_from_label_text(line: str):
#     vs = VehicleState('car')
#     vs.read_from_line(line.split())
#     return vs


# def get_objects_from_label_file():
#     raise

# def get_object_from_label_text(label_text: str):
#     label_items = [item.rstrip(',') for item in label_text.split()]
#     form = label_items[0]
#     if form.lower() == 'kitti':
#         raise NotImplementedError
#         # label_items.insert(1, 'track_3d')
#         # label_items.insert(15, 'CameraCoordinates')
#         # label = get_avstack_object_from_label_items(label_items)
#     elif form.lower() == 'avstack':
#         label = get_avstack_object_from_label_items(label_items)
#     elif form.lower() == 'nuscenes':
#         raise NotImplementedError(form)
#     else:
#         raise NotImplementedError(form)
#     return label


# def get_avstack_object_from_label_items(label_items: list):
#     frame = int(label_items[2])
#     timestamp = float(label_items[3])
#     ID = int(label_items[4])
#     obj_type = str(label_items[5])
#     if label_items[1] == 'track_2d':
#         raise NotImplementedError
#     elif label_items[1] == 'track_3d':
#         pos = [float(d) for d in label_items[6:9]]
#         vel = [float(d) for d in label_items[9:12]]
#         acc = [float(d) for d in label_items[12:15]]
#         h, w, l, yaw = [float(d) for d in label_items[15:19]]
#         score = float(label_items[19])
#         coordinates = get_coordinates_from_string(label_items[20])
#         box3d = Box3D([h,w,l,pos,yaw], coordinates)
#         label = VS.set(frame, timestamp, obj_type, pos, vel, acc, yaw, box3d, coordinates)
#     elif label_items[1] == 'track_3d2d':
#         raise NotImplementedError
#     else:
#         raise NotImplementedError(label_items[1])
#     return label


# def get_objects_from_label_file(label_file, whitelist_types='all', ignore_types=[]):
#     assert os.path.exists(label_file), f'Cannot find requested label file at {label_file}'
#     with open(label_file, 'r') as f:
#         lines = [line.rstrip() for line in f]
#     labels = [get_object_from_label_text(line) for line in lines]
#     if (whitelist_types != 'all') and ('all' not in whitelist_types):
#         labels = [lab for lab in labels if (lab.obj_type not in ignore_types) & (lab.obj_type in whitelist_types)]
#     return np.asarray(labels)


# def get_object_from_label_text(label_text: str):
#     label_items = [item.rstrip(',') for item in label_text.split()]
#     form = label_items[0]
#     if form.lower() == 'kitti':
#         # remove unnecessary items
#         label_items.pop(4)  # alpha
#         label_items.pop(3)  # occlusion
#         label_items.pop(2)  # truncation
#         # add relevant items
#         label_items.insert(1, 'object_3d2d')
#         label_items.insert(2, 'lidar')
#         label_items.insert(15, 'CameraCoordinates')
#         label = get_avstack_object_from_label_items(label_items)
#     elif form.lower() == 'avstack':
#         label = get_avstack_object_from_label_items(label_items)
#     elif form.lower() == 'nuscenes':
#         raise NotImplementedError(form)
#     else:
#         raise NotImplementedError(form)
#     return label


# def get_avstack_object_from_label_items(label_items: list):
#     """
#     Expected format:

#     2D Label:
#     dataset, label_type, sensor, obj_type, 2dxmin, 2dymin, 2dxmax, 2dymax

#     3D Label
#     dataset, label_type, sensor, obj_type, h, w, l, tx, ty, tz, yaw, coords

#     3D2D Label:
#     dataset, label_type, sensor, obj_type, 2dxmin, 2dymin, 2dxmax, 2dymax, h, w, l, tx, ty, tz, yaw, coords
#     """
#     sensor_source = label_items[2]
#     obj_type = label_items[3]
#     if label_items[1] == 'object_2d':
#         box2d = [float(d) for d in label_items[4:8]]  # 4 elements for box 2d
#         box2d = Box2D(box2d)
#         label = Object2dLabel(sensor_source, obj_type, box2d)
#     elif label_items[1] == 'object_3d':
#         box3d = [float(d) for d in label_items[4:11]]  # 7 elements for box 3d
#         coords = label_items[11]  # coordinate frame of the 3d box
#         box3d = Box3D(box3d, get_coordinates_from_string(coords))
#         label = Object3dLabel(sensor_source, obj_type, box3d)
#     elif label_items[1] == 'object_3d2d':
#         box2d = [float(d) for d in label_items[4:8]]  # 4 elements for box 2d
#         box3d = [float(d) for d in label_items[8:15]]  # 7 elements for box 3d
#         coords = label_items[15]
#         box2d = Box2D(box2d)
#         box3d = Box3D(box3d, get_coordinates_from_string(coords))
#         label = Object3d2dLabel(sensor_source, obj_type, box2d, box3d)
#     else:
#         raise NotImplementedError(label_items[1])
#     return label


# def get_avstack_label_text_from_2d_box(obj_type, sensor_source, box_2d, score):
#     return f'avstack, object_2d, {sensor_source}, {obj_type}, {box_2d.xmin}, ' + \
#            f'{box_2d.ymin}, {box_2d.xmax}, {box_2d.ymax}, {score}'


# def get_avstack_label_text_from_3d_box(obj_type, sensor_source, box_3d, score):
#     return f'avstack, object_3d, {sensor_source}, {obj_type},  {box_3d.h}, ' + \
#            f'{box_3d.w}, {box_3d.l},  {box_3d.t[0]}, {box_3d.t[1]}, ' + \
#            f'{box_3d.t[2]}, {box_3d.yaw}, {str(box_3d.coordinates)} {score}'