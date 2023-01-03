import os
import numpy as np
import quaternion
from cv2 import imread, imwrite


from .dataset import BaseSceneDataset, BaseSceneManager

from avstack import calibration
from avstack.objects import Occlusion
from avstack.geometry import Origin, NominalOriginStandard
from avstack.geometry import R_stan_to_cam, R_cam_to_stan
from avstack.utils import maskfilters
from avstack import transformations as tforms
    

general_to_detection_class = \
    {'animal':'ignore',
     'human.pedestrian.personal_mobility':   'ignore',
     'human.pedestrian.stroller':            'ignore',
     'human.pedestrian.wheelchair':          'ignore',
     'movable_object.debris':                'ignore',
     'movable_object.pushable_pullable':     'ignore',
     'static_object.bicycle_rack':           'ignore',
     'vehicle.emergency.ambulance':          'ignore',
     'vehicle.emergency.police':             'ignore',
     'movable_object.barrier':               'barrier',
     'vehicle.bicycle':                      'bicycle',
     'vehicle.bus.bendy':                    'bus',
     'vehicle.bus.rigid':                    'bus',
     'vehicle.car':                          'car',
     'vehicle.construction':                 'construction_vehicle',
     'vehicle.motorcycle':                   'motorcycle',
     'human.pedestrian.adult':               'pedestrian',
     'human.pedestrian.child':               'pedestrian',
     'human.pedestrian.construction_worker': 'pedestrian',
     'human.pedestrian.police_officer':      'pedestrian',
     'movable_object.trafficcone':           'traffic_cone',
     'vehicle.trailer':                      'trailer',
     'vehicle.truck':                        'truck'}
_nominal_whitelist_types = ['car', 'pedestrian', 'bicycle',
        'truck', 'bus', 'motorcycle']
_nominal_ignore_types = []


class _nuManager(BaseSceneDataset):
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(self, nuX, nuX_can, data_dir, split='v1.0-mini', verbose=False):
        self.nuX = nuX
        self.nuX_can = nuX_can
        self.data_dir = data_dir
        self.split = split
        self.split_path = os.path.join(data_dir, split)


class _nuBaseDataset(BaseSceneDataset):
    nominal_whitelist_types = ['car', 'pedestrian', 'bicycle',
        'truck', 'bus', 'motorcycle']
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(self, nuX, nuX_can, data_dir, split, verbose=False,
            whitelist_types=_nominal_whitelist_types, ignore_types=nominal_ignore_types):
        super().__init__(whitelist_types, ignore_types)
        self.nuX = nuX
        self.nuX_can = nuX_can
        self.data_dir = data_dir
        self.split = split
        self.split_path = os.path.join(data_dir, split)
        self.make_sample_records()

    def __str__(self):
        return f'{self.NAME} ObjectDataset of folder: {self.split_path}'

    @property
    def frames(self):
        return list(self.sample_records.keys())

    def make_sample_records(self):
        raise NotImplementedError

    def _get_sample_metadata(self, sample_token):
        return self.nuX.get('sample', sample_token)

    def _get_anns_metadata(self, sample_data_token):
        try:
            _, boxes, camera_intrinsic = self.nuX.get_sample_data(sample_data_token)
            boxes = [{'name':box.name, 'box':box} for box in boxes]
        except AttributeError:
            object_tokens, surface_tokens = self.nuX.list_anns(sample_data_token, verbose=False)
            obj_anns = [self.nuX.get('object_ann', obj_tok) for obj_tok in object_tokens]
            boxes = [{'name':self.nuX.get('category', obj_ann['category_token'])['name'],
                      'box':obj_ann['bbox']} for obj_ann in obj_anns]
        return boxes

    def _get_sensor_record(self, frame, sensor=None):
        try:
            sr = self.nuX.get('sample_data', self.sample_records[frame]['data'][sensor.upper()])
        except KeyError:
            sr = self.nuX.get('sample_data', self.sample_records[frame]['key_camera_token'])
        return sr

    def _get_calib_data(self, frame, sensor): 
        try:
            calib_data = self.nuX.get('calibrated_sensor',
                self._get_sensor_record(frame, sensor)['calibrated_sensor_token'])
        except KeyError as e:
            try:
                calib_data = self.nuX.get('calibrated_sensor',
                    self._get_sensor_record(frame, self.sensors[sensor])['calibrated_sensor_token'])
            except KeyError as e2:
                raise e  # raise the first exception
        return calib_data

    def _get_sensor_file_name(self, frame, sensor=None):
        return os.path.join(self.data_dir, self._get_sensor_record(frame, sensor)['filename'])

    def _load_frames(self, sensor: str=None):
        return self.frames

    def _load_calibration(self, frame, sensor=None):
        """
        Reference frame has standard coordinates:
            x: forward
            y: left
            z: up

        reference frame is: "the center of the rear axle projected to the ground."
        """
        calib_data = self._get_calib_data(frame, sensor)
        x_O_2_S_in_O = np.array(calib_data['translation'])
        q_O_to_S = np.quaternion(*calib_data['rotation']).conjugate()
        origin = Origin(x_O_2_S_in_O, q_O_to_S)
        sensor = sensor if sensor is not None else self.sensor_name(frame)
        if 'CAM' in sensor:
            P = np.hstack((np.array(calib_data['camera_intrinsic']), np.zeros((3,1))))
            calib = calibration.CameraCalibration(origin, P, self.img_shape)
        else:
            calib = calibration.Calibration(origin)
        return calib

    def _load_timestamp(self, frame, sensor):
        return self._get_sensor_record(frame, sensor)['timestamp']/1e6 - self.t0

    def _load_image(self, frame, sensor=None):
        img_fname = self._get_sensor_file_name(frame, sensor)
        return imread(img_fname)[:,:,::-1]  # convert to RGB

    def _load_ego(self, frame, sensor='LIDAR_TOP'):
        sd_record = self._get_sensor_record(frame, sensor)
        ego_data = self.nuX.get('ego_pose', sd_record['ego_pose_token'])
        ts = ego_data['timestamp']/1e6 - self.t0
        line = self._ego_to_line(ts, ego_data)
        return self.parse_label_line(line)

    def _load_objects(self, frame, sensor=None, whitelist_types=['car', 'pedestrian', 'bicycle',
            'truck', 'bus', 'motorcycle'], ignore_types=[]):
        """
        automatically loads into local sensor coordinates

        class_names = [
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
        """
        sensor_data = self._get_sensor_record(frame, sensor)
        origin = self.get_calibration(frame, sensor).origin
        try:
            boxes = self._get_anns_metadata(sensor_data['token'])
        except KeyError:
            boxes = self._get_anns_metadata(sensor_data['sample_token'])
        objects = []
        for box in boxes:
            obj_type = general_to_detection_class[box['name']]
            if (obj_type in whitelist_types) or (whitelist_types=='all') or ('all' in whitelist_types):
                ts = sensor_data['timestamp']
                line = self._box_to_line(ts, box['box'], origin)
                obj = self.parse_label_line(line)
                objects.append(obj)
        return np.array(objects)

    def _box_to_line(self, ts, box, origin):
        ID = box.token
        obj_type = general_to_detection_class[box.name]
        vel = [None]*3
        acc = [None]*3
        w, l, h = box.wlh
        pos = box.center
        q = box.orientation
        occ = Occlusion.UNKNOWN
        line = f'nuscenes object_3d {ts} {ID} {obj_type} {int(occ)} {pos[0]} ' \
               f'{pos[1]} {pos[2]} {vel[0]} {vel[1]} {vel[2]} {acc[0]} ' \
               f'{acc[1]} {acc[2]} {h} {w} {l} {q[0]} {q[1]} {q[2]} {q[3]} {"center"} {origin.format_as_string()}'
        return line

    def _ego_to_line(self, ts, ego):
        obj_type = 'car'
        ID = ego['token']
        pos = np.array(ego['translation'])
        q = np.quaternion(*ego['rotation']).conjugate()
        if self.ego_speed_interp is not None:
            vel = tforms.transform_orientation(q, 'quat', 'dcm')[:,0] * self.ego_speed_interp(ts)
        else:
            vel = [None]*3
        acc = [None]*3
        w, l, h = [1.73, 4.084, 1.562]
        origin = NominalOriginStandard
        occ = Occlusion.NONE
        line = f'nuscenes object_3d {ts} {ID} {obj_type} {int(occ)} {pos[0]} ' \
               f'{pos[1]} {pos[2]} {vel[0]} {vel[1]} {vel[2]} {acc[0]} ' \
               f'{acc[1]} {acc[2]} {h} {w} {l} {q.w} {q.x} {q.y} {q.z} {"center"} {origin.format_as_string()}'
        return line
