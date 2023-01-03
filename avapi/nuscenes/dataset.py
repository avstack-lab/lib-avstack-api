# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-09-05
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-29
# @Description:
"""

"""
import os, logging
import numpy as np
from scipy.interpolate import interp1d
from .._nu_base import _nominal_ignore_types, _nominal_whitelist_types, _nuManager, _nuBaseDataset
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.utils.splits import create_splits_scenes as create_splits_scenes_nusc
    splits_scenes = create_splits_scenes_nusc()
except ModuleNotFoundError as e:
    print('Cannot import nuscenes')
    splits_scenes = None


class nuScenesManager(_nuManager):
    name = 'nuScenes'

    def __init__(self, data_dir, split='v1.0-mini', verbose=False):
        nusc = NuScenes(version=split, dataroot=data_dir, verbose=verbose)
        try:
            nusc_can = NuScenesCanBus(dataroot=data_dir)
        except Exception as e:
            logging.warning('Cannot find CAN bus data')
            nusc_can = None
        super().__init__(nusc, nusc_can, data_dir, split, verbose)
        self.scene_name_to_index = {}
        self.scene_number_to_index = {}
        self.index_to_scene = {}
        self.scenes = nusc.scene
        for i, sc in enumerate(nusc.scene):
            self.scene_name_to_index[sc['name']] = i
            self.scene_number_to_index[int(sc['name'].replace('scene-',''))] = i
            self.index_to_scene[i] = sc['name']

    def __len__(self):
        return len(self.scenes)

    def list_scenes(self):
        self.nuX.list_scenes()

    def get_scene_dataset_by_name(self, scene_name):
        idx = self.scene_name_to_index[scene_name]
        return self.get_scene_dataset_by_index(idx)

    def get_scene_dataset_by_scene_number(self, scene_number):
        idx = self.scene_number_to_index[scene_number]
        return self.get_scene_dataset_by_index(idx)

    def get_scene_dataset_by_index(self, scene_idx):
        return nuScenesSceneDataset(self.data_dir, self.split,
            self.nuX.scene[scene_idx], nusc=self.nuX, nusc_can=self.nuX_can)


class nuScenesSceneDataset(_nuBaseDataset):
    NAME = 'nuScenes'
    CFG = {}
    CFG['num_lidar_features'] = 5
    CFG['IMAGE_WIDTH'] = 1600
    CFG['IMAGE_HEIGHT'] = 900
    CFG['IMAGE_CHANNEL'] = 3
    img_shape = (CFG['IMAGE_HEIGHT'], CFG['IMAGE_WIDTH'], CFG['IMAGE_CHANNEL'])
    sensors = {'lidar':'LIDAR_TOP', 'main_lidar':'LIDAR_TOP', 'main_camera':'CAM_FRONT'}
    keyframerate = 2
    sensor_IDs = {'CAM_BACK':3, 'CAM_BACK_LEFT':1, 'CAM_BACK_RIGHT':2,
                  'CAM_FRONT':0, 'CAM_FRONT_LEFT':4, 'CAM_FRONT_RIGHT':5,
                  'LIDAR_TOP':0, 'RADAR_BACK_LEFT':0, 'RADAR_BACK_RIGHT':1,
                  'RADAR_FRONT':2, 'RADAR_FRONT_LEFT':3, 'RADAR_FRONT_RIGHT':4}

    def __init__(self, data_dir, split, scene, nusc=None, nusc_can=None, verbose=False,
            whitelist_types=_nominal_whitelist_types, ignore_types=_nominal_ignore_types):
        nusc = nusc if nusc is not None else NuScenes(
            version=split, dataroot=data_dir, verbose=verbose)
        try:
            nusc_can = nusc_can if nusc_can is not None else NuScenesCanBus(dataroot=data_dir)
        except Exception as e:
            logging.warning('Cannot find CAN bus data')
            nusc_can = None
        self.scene = scene
        self.scene_name = self.scene
        self.framerate = 2
        self.sequence_id = scene['name']
        self.splits_scenes = splits_scenes
        try:
            veh_speed = self.nuX_can.get_messages(self.scene['name'], 'vehicle_monitor')
        except Exception as e:
            self.ego_speed_interp = None
        else:
            veh_speed = np.array([(m['utime'], m['vehicle_speed']) for m in veh_speed])
            veh_speed[:, 1] *= 1 / 3.6
            self.ego_speed_interp = interp1d(veh_speed[:,0]/1e6-self.t0, veh_speed[:,1], fill_value='extrapolate')
        super().__init__(nusc, nusc_can, data_dir, split, verbose, whitelist_types, ignore_types)

    def make_sample_records(self):
        self.sample_records = {0:self.nuX.get('sample', self.scene['first_sample_token'])}
        for i in range(1, self.scene['nbr_samples'], 1):
            self.sample_records[i] = self.nuX.get('sample', self.sample_records[i-1]['next'])
        self.t0 = self.sample_records[0]['timestamp'] / 1e6

    def _load_lidar(self, frame, sensor='LIDAR_TOP', filter_front=False, with_panoptic=False):
        if sensor.lower() == 'lidar':
            sensor = self.sensors['lidar']
        lidar_fname = self._get_sensor_file_name(frame, sensor)
        lidar = np.fromfile(lidar_fname, dtype=np.float32).reshape((-1, self.CFG['num_lidar_features']))
        if with_panoptic:
            lidar = np.concatenate((lidar, self._load_panoptic_lidar_labels(frame, sensor)[:,None]), axis=1)
        if filter_front:
            return lidar[lidar[:,1]>0,:]  # y is front on nuscenes
        else:
            return lidar

    def _load_panoptic_lidar_labels(self, frame, sensor='LIDAR_TOP'):
        record = self.nuX.get('panoptic', self._get_sensor_record(frame, sensor)['token'])
        fname = os.path.join(self.data_dir, record['filename'])
        panoptic_labels = (np.load(fname)['data'] // 1000).astype(int)
        return panoptic_labels