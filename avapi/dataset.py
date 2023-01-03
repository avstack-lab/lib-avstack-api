# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-08-08
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-09-29
# @Description:
"""

"""
import hashlib
import os, sys, glob
import numpy as np
import quaternion
from avstack import sensors, calibration
from avstack import transformations as tforms
from avstack.geometry import bbox, get_origin_from_line, \
    q_cam_to_stan, q_stan_to_cam, NominalOriginStandard
from avstack.objects import VehicleState, Occlusion



def wrap_minus_pi_to_pi(phases):
    phases = (phases + np.pi) % (2 * np.pi) - np.pi
    return phases


class BaseSceneManager():
    def get_splits_scenes(self):
        return self.splits_scenes

    def make_splits_scenes(self, modval=4, seed=1):
        """Split the scenes by hashing the experiment name and modding
        3:1 split using mod 4
        """
        np.random.seed(seed)
        splits_scenes = {'train':[], 'val':[]}
        for scene in self.scenes:
            vh = 3*int(hashlib.sha1(scene.encode("utf-8")).hexdigest(), 16)
            v = vh % modval
            if v == (modval-1):
                splits_scenes['val'].append(scene)
            else:
                splits_scenes['train'].append(scene)
        return splits_scenes


class BaseSceneDataset():
    sensor_IDs = {}
    def __init__(self, whitelist_types, ignore_types):
        self.whitelist_types = whitelist_types
        self.ignore_types = ignore_types

    def __repr__(self):
        return self.__str__()

    @property
    def frames(self):
        raise NotImplementedError

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
        return self._load_calibration(frame, sensor)

    def get_ego(self, frame):
        return self._load_ego(frame)

    def get_image(self, frame, sensor):
        sensor = self.get_sensor_name(sensor)
        ts = self.get_timestamp(frame, sensor)
        data = self._load_image(frame, sensor=sensor)
        cam_string = 'image-%i'%sensor if isinstance(sensor, int) else sensor
        cam_string = cam_string if sensor is not None else self.sensor_name(frame)
        calib = self.get_calibration(frame, cam_string)
        return sensors.ImageData(ts, frame, data, calib, self.get_sensor_ID(cam_string))

    def get_sensor_data_filepath(self, frame, sensor):
        sensor = self.get_sensor_name(sensor)
        return self._load_sensor_data_filepath(frame, sensor)

    def get_depthimage(self, frame, sensor):
        sensor = self.get_sensor_name(sensor)
        ts = self.get_timestamp(frame, sensor)
        data = self._load_image(frame, sensor=sensor)
        cam_string = 'image-%i'%sensor if isinstance(sensor, int) else sensor
        calib = self.get_calibration(frame, cam_string)
        return sensors.DepthImageData(ts, frame, data, calib, self.get_sensor_ID(cam_string))

    def get_lidar(self, frame, sensor=None, filter_front=False, min_range=None,
            max_range=None, with_panoptic=False):
        if sensor is None:
            sensor = self.sensors['lidar']
        sensor = self.get_sensor_name(sensor)
        ts = self.get_timestamp(frame, sensor)
        data = self._load_lidar(frame, sensor, filter_front=filter_front, with_panoptic=with_panoptic)
        calib = self.get_calibration(frame, sensor)
        pc = sensors.LidarData(ts, frame, data, calib, self.get_sensor_ID(sensor))
        if (min_range is not None) or (max_range is not None):
            pc.filter_by_range(min_range, max_range, inplace=True)
        return pc

    def get_objects(self, frame, sensor=None, max_dist=None, max_occ=None, **kwargs):
        sensor = self.get_sensor_name(sensor)
        objs = self._load_objects(frame, sensor=sensor, **kwargs)
        if max_occ is not None:
            objs = np.array([obj for obj in objs if (obj.occlusion <= max_occ) or (obj.occlusion == Occlusion.UNKNOWN)])
        if max_dist is not None:
            if sensor == 'ego':
                calib = calibration.Calibration(NominalOriginStandard)
            else:
                calib = self.get_calibration(frame, sensor)
            objs = np.array([obj for obj in objs if obj.position.distance(calib.origin)<max_dist])
        return objs

    def get_objects_global(self, frame, max_dist=None, **kwargs):
        return self._load_objects_global(frame, **kwargs)

    def get_objects_from_file(self, fname, whitelist_types, max_dist=None):
        return self._load_objects_from_file(fname, whitelist_types, max_dist=max_dist)

    def get_timestamp(self, frame, sensor='lidar'):
        sensor = self.get_sensor_name(sensor)
        return self._load_timestamp(frame, sensor)

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

    def _load_calibration(self, frame, sensor):
        raise NotImplementedError

    def _load_image(self, frame, camera):
        raise NotImplementedError

    def _load_lidar(self, frame):
        raise NotImplementedError

    def _load_objects(self, frame, sensor):
        raise NotImplementedError

    def _load_sensor_data_filepath(self, frame, sensor: str):
        return self._get_sensor_file_name(frame, sensor)

    def _load_objects_from_file(self, fname, whitelist_types=None, ignore_types=None,
            max_dist=None, dist_ref=NominalOriginStandard):
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
        with open(fname, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]
        objects = []
        for line in lines:
            obj = self.parse_label_line(line)
            # -- type filter
            if ('all' in whitelist_types) or (obj.obj_type.lower() in whitelist_types):
                if obj.obj_type.lower() in ignore_types:
                    continue
            else:
                continue
            # -- distance filter
            if max_dist is not None:
                obj.change_origin(dist_ref)
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
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def parse_label_line(self, label_file_line):
        # Parse data elements
        data = label_file_line.strip('\n').split(' ')
        if data[0] in ['avstack', 'nuscenes']:
            idx = 2
            ts = data[idx]; idx += 1
            ID = data[idx]; idx += 1
            obj_type = data[idx]; idx += 1
            occ = Occlusion(int(data[idx])); idx += 1
            pos = data[idx:(idx+3)]; idx += 3
            t_box = pos
            vel = data[idx:(idx+3)]; idx += 3
            if np.all([v=='None' for v in vel]):
                vel = None
            acc = data[idx:(idx+3)]; idx += 3
            if np.all([a=='None' for a in acc]):
                acc = None
            box_size = data[idx:(idx+3)]; idx += 3
            q_O_to_obj = np.quaternion(*[float(d) for d in data[idx:(idx+4)]]); idx+=4
            if data[0] == 'nuscenes':
                q_O_to_obj = q_O_to_obj.conjugate()
            ang = None
            where_is_t = data[idx]; idx += 1
            object_origin = get_origin_from_line(' '.join(data[idx:]))
        elif data[0] == 'kitti-v2':  # converted kitti raw data -- longitudinal dataset
            idx = 1
            ts = data[idx]; idx += 1
            ID = data[idx]; idx += 1
            obj_type = data[idx]; idx += 1
            orientation = data[idx]; idx += 1
            occ = Occlusion.UNKNOWN
            box2d = data[idx:(idx+4)]; idx += 4
            box_size = data[idx:(idx+3)]; idx += 3
            t_box = data[idx:(idx+3)]; idx += 3
            yaw = float(data[idx]); idx += 1
            score = float(data[idx]); idx += 1
            object_origin = get_origin_from_line(' '.join(data[idx:]))
            pos = t_box
            vel = acc = ang = None
            where_is_t = 'bottom'
            q_V_to_obj = tforms.transform_orientation([0,0,yaw], 'euler', 'quat')
            q_O_to_V = object_origin.q.conjugate()
            q_O_to_obj = q_V_to_obj * q_O_to_V

        else:  # assume kitti with no prefix -- this is for kitti static dataset
            ts = 0.0
            ID = np.random.randint(low=0, high=1e6)  # not ideal but ensures unique IDs
            obj_type = data[0]
            occ = Occlusion.UNKNOWN
            box2d = data[4:8]
            t_box = data[11:14]  # x_C_2_Obj == x_O_2_Obj for O as camera
            where_is_t = 'bottom'
            box_size = data[8:11]
            pos = t_box
            vel = acc = ang = None
            yaw = -float(data[14])-np.pi/2
            object_origin = self.get_calibration(self.frames[0], 'image-2').origin
            q_Ccam_to_Cstan = q_cam_to_stan
            q_Cstan_to_obj = tforms.transform_orientation([0,0,yaw], 'euler', 'quat')
            q_O_to_obj = q_Cstan_to_obj * q_Ccam_to_Cstan

        # Put into objects
        t_box = np.array([float(t) for t in t_box])
        box3d = bbox.Box3D([*[float(b) for b in box_size], t_box, q_O_to_obj], object_origin, where_is_t=where_is_t)
        pos = [float(p) for p in pos]
        vel = [float(v) for v in vel] if vel is not None else None
        acc = [float(a) for a in acc] if acc is not None else None
        att = box3d.q
        ang = [float(a) for a in ang] if ang is not None else None
        try:
            ID = int(ID)
        except ValueError as e:
            pass
        obj = VehicleState(obj_type, ID)
        obj.set(float(ts), pos, box3d, vel, acc, att, ang, occlusion=occ, origin=object_origin)
        return obj