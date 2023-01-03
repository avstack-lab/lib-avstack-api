# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-04
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-28
# @Description:
"""

"""


import os
import shutil
import pickle
from datetime import datetime
import numpy as np
import avstack.transformations as tforms
from avstack.geometry import Translation, Vector, Rotation, Transform
from avstack.geometry import CarlaCoordinates, CameraCoordinates, StandardCoordinates
from avstack import sensors
from avapi.carla import utils


# =================================================
# TRUTH DATA
# =================================================

class CarlaTruthRecorder():
    """Records truth data from the world in avstack format"""
    def __init__(self, save_folder, format_as=['avstack']):
        self.folder = save_folder
        self.format_as = format_as
        assert isinstance(format_as, list)
        self.format_folders = []
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        for form in format_as:
            self.format_folders.append(os.path.join(self.folder, 'objects', form))
            os.makedirs(self.format_folders[-1], exist_ok=False)

    def restart(self, save_folder):
        self.__init__(save_folder, self.format_as)

    def record(self, t, frame, ego, npcs):
        """Main call to log data"""
        # -- record object data
        ego_data = utils.wrap_actor_to_vehicle_state(t, ego.actor)
        npc_data = [utils.wrap_actor_to_vehicle_state(t, npc) for npc in npcs]
        self._save_object_data(t, frame, [ego_data], 'ego')
        self._save_object_data(t, frame, npc_data, 'npc')
        return True  # if successful

    def _save_object_data(self, timestamp, frame, data, suffix):
        file = 'timestamp_%08.2f-frame_%06d-%s.{}' % (timestamp, frame, suffix)
        for folder, form in zip(self.format_folders, self.format_as):
            if form == 'pickle':
                pickle.dump(data, open(os.path.join(folder, file.format('p')), 'wb'))
            elif form in ['kitti-object', 'apolloscape-trajectory', 'avstack']:
                data_strs = "\n".join([d.format_as(form) for d in data])
                with open(os.path.join(folder, file.format('txt')), 'w') as f:
                    f.write(data_strs)
            else:
                raise NotImplementedError(f'Format {form} not implemented')

    def _save_lane_lines(self):
        raise NotImplementedError


# =================================================
# REAL-TIME DATA
# =================================================

class CarlaDataRecorder():
    """Records data from each of the modules"""
    def __init__(self, ego):
        self._ego = ego
        raise NotImplementedError

    def log(self):
        """Main call to log data"""
        self._record_localization()
        self._record_perception()
        self._record_tracking()
        self._record_prediction()
        self._record_planning()
        self._record_control()
        return True  # if successful
