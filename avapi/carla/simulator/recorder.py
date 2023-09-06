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

from avapi.carla.simulator import utils


# =================================================
# TRUTH DATA
# =================================================


class CarlaTruthRecorder:
    """Records truth data from the world in avstack format"""

    def __init__(self, save_folder, remove_if_exist=True):
        self.remove_if_exist = remove_if_exist
        self.save_folder = save_folder
        self.folder = os.path.join(save_folder, "objects")
        if remove_if_exist and os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(self.folder, exist_ok=True)

    def restart(self, save_folder):
        self.__init__(save_folder, remove_if_exist=self.remove_if_exist)

    def record(self, t, frame, ego, npcs):
        """Main call to log data"""
        # -- record object data
        ego_data = utils.wrap_actor_to_vehicle_state(t, ego.actor)
        npc_data = [utils.wrap_actor_to_vehicle_state(t, npc) for npc in npcs]
        self._save_object_data(t, frame, [ego_data], "ego")
        self._save_object_data(t, frame, npc_data, "npc")
        return True  # if successful

    def _save_object_data(self, timestamp, frame, data, suffix):
        file = "timestamp_%08.2f-frame_%06d-%s.{}" % (timestamp, frame, suffix)
        data_strs = "\n".join([d.encode() for d in data])
        with open(os.path.join(self.folder, file.format("txt")), "w") as f:
            f.write(data_strs)

    def _save_lane_lines(self):
        raise NotImplementedError


# =================================================
# REAL-TIME DATA
# =================================================


class CarlaDataRecorder:
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
