# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-29
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-11
# @Description:
"""

"""

import os
import glob
from avapi.evaluation import ResultManager
from avapi.visualize import create_track_movie


# ========================================================
# BASE REPLAY HELPERS
# ========================================================

class _VideoReplay():
    def __init__(self, SM):
        """
        :SM - Scene Manager object
        """
        self.SM = SM

    def _callback(self, idx):
        pass

    def compile(self):
        raise NotImplementedError

    def show(self, extent, inline=True):
        raise NotImplementedError


class ObjectVideoReplay(_VideoReplay):
    def __init__(self, SM):
        super().__init__(SM)
        self.objects = {'truths':[], 'detections':[], 'tracks':[]}

    def add_objects_from_track_results(self, track_results):
        """Add both truths and tracks from track result class"""
        pass

    def add_objects_from_percep_results(self, percep_results):
        """Add both detections and truths from percep result class"""
        pass

    def add_object(self, object, color, identifier):
        """Add objects of a particular type"""
        # -- check existing objects so we don't have exact duplicates (?)
        

        # -- add object
        self.objects[identifier].append({'object':object, 'color':color})


    def compile(self):
        """Compile video frames ahead of time"""

    def show(self, extent, inline=True):
        """Display video"""


def load_ground_truth_data(folder):
    glob_dir = glob.glob(os.path.join(folder, "*.txt"))
    glob_dir = sorted(glob_dir)
    ego_data = []
    npc_data = []
    for file in glob_dir:
        f_info = file.split('/')[-1].split('-')
        timestamp = float(f_info[0].split('_')[1])
        frame = int(f_info[1].split('_')[1])
        obj = f_info[2].replace('.txt','')
        with open(file, 'r') as f:
            lines = [line.rstrip() for line in f]
        if obj == 'ego':
            assert len(lines) == 1
            ego_data.append(get_object_from_label_text(lines[0]))
        elif obj == 'npc':
            npc_data.append([get_object_from_label_text(l) for l in lines])
        else:
            raise NotImplementedError(obj)
    return ego_data, npc_data


def replay_ground_truth(folder):
    print('Replaying ground truth data from {}'.format(folder))
    ego_data, npc_data = load_ground_truth_data(folder)

    # -- ego-centric frame
    new_ego = []
    new_npcs = []
    for ego, npcs in zip(ego_data, npc_data):
        new_npcs.append([ego.global_to_local(npc) for npc in npcs])
        new_ego.append(ego.global_to_local(ego))

    # -- make track results class
    object_results = {i:ResultManager(npcs, []) for i, npcs in enumerate(new_npcs)}

    # -- visualizer
    extent = [(0,60), (-15, 15), (-5, 5)]
    ego_box = new_ego[0].box3d
    create_track_movie(extent, object_results, ego_box=ego_box, inline=False)


# ========================================================
# BASE REPLAY HELPERS
# ========================================================
