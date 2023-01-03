# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-09-05
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-10-22
# @Description:
"""
Take objects in the global frame and create their local
representations in each of the sensors
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from copy import copy, deepcopy
from multiprocessing import Pool
from functools import partial
import logging

import avstack
import avapi

from avstack.objects import Occlusion


def main(args, frame_start=4, frame_end_trim=4, n_frames_max=10000):
    CSM = avapi.carla.CarlaSceneManager(args.data_dir)
    for CDM in tqdm(CSM):
        with_multi = True
        chunksize = 10
        frames = [f for f in list(CDM.ego_frame_to_ts.keys()) if f >= frame_start]
        frames = frames[:max(1, min(n_frames_max, len(frames))-frame_end_trim)]
        frames_all = frames
        egos = {i:CDM.get_ego(i) for i in frames}
        nproc = max(1, min(100, int(len(frames)/chunksize)))
        if with_multi:
            print('Getting global objects from all frames')
            with Pool(nproc) as p:
                part_func = partial(get_obj_glob_by_frames, CDM)
                objects_global = dict(zip(frames, tqdm(p.imap(part_func,
                    frames, chunksize=chunksize), position=0, leave=True, total=len(frames))))
            print('Projecting objects into ego frame')
            with Pool(nproc) as p:
                part_func = partial(get_obj_ego_by_frames, CDM)
                objects_ego = dict(zip(frames, tqdm(p.istarmap(part_func,
                    zip(egos.values(), objects_global.values()), chunksize=chunksize),
                    position=0, leave=True, total=len(frames))))
        else:
            print('Getting global objects from all frames')
            objects_global = {i_frame:get_obj_glob_by_frames(CDM, i_frame) for i_frame in tqdm(frames)}
            print('Projecting objects into ego frame')
            objects_ego = {i_frame:get_obj_ego_by_frames(CDM, ego, objs_global_this) for (i_frame, ego), objs_global_this in tqdm(zip(egos.items(), objects_global.values()), total=len(frames))}

        print('Processing ego')
        process_func_sensors(CDM, 'ego', egos, objects_ego, frames, args.data_dir)

        print('Looping over sensors')
        for sens, frames in CDM.sensor_frames.items():
            frames_this = [frame for frame in frames if frame in frames_all]
            egos_this = {frame:egos[frame] for frame in frames_this}
            objs_ego_this = {frame:objects_ego[frame] for frame in frames_this}
            process_func_sensors(CDM, sens, egos_this, objs_ego_this, frames_this, args.data_dir)


def get_obj_glob_by_frames(CDM, i_frame):
    return CDM.get_objects_global(i_frame)


def get_obj_ego_by_frames(CDM, ego, objects_global_this):
    return [ego.global_to_local(obj) for obj in objects_global_this]


def process_func_sensors(CDM, sens, egos, objects_ego, frames, data_dir):
    """
    Post-process frames for a sensor
    """
    assert len(egos) == len(objects_ego) == len(frames), '{}, {}, {}'.format(len(egos), len(objects_ego), len(frames))
    obj_sens_folder = os.path.join(data_dir, CDM.scene, 'objects_sensor', sens)
    os.makedirs(obj_sens_folder, exist_ok=True)
    func = partial(process_func_frames, CDM, sens, obj_sens_folder)

    with_multi = True
    chunksize = 20
    nproc = max(1, min(20, int(len(frames)/chunksize)))
    if with_multi:
        with Pool(nproc) as p:
            res = list(tqdm(p.istarmap(func,
                zip(egos.values(), objects_ego.values(), frames), chunksize=chunksize),
                position=0, leave=True, total=len(frames)))
    else:
        for i_frame in tqdm(frames):
            func(egos[i_frame], objects_ego[i_frame], i_frame)


def process_func_frames(CDM, sens, obj_sens_folder, ego, objects_ego, i_frame):
    # -- get objects local to ego
    if sens == 'ego':
        objs_sens = objects_ego
    else:
        try:
            calib = CDM.get_calibration(i_frame, sens)
        except FileNotFoundError as e:
            if i_frame > 10:
                return  # probably just because we stopped early (?)
            else:
                raise e

        # -- add ego object to objects if other sensor
        if np.linalg.norm(calib.origin.x) > 5:
            ego_copy = deepcopy(ego)
            objects_ego.append(ego.global_to_local(ego_copy))

        # -- change to sensor origin
        for obj in objects_ego:
            obj.change_origin(calib.origin)

        # -- filter in view of sensor
        if 'cam' in sens.lower():
            objs_sens = [obj for obj in objects_ego if
                avstack.utils.maskfilters.box_in_fov(obj.box, calib,
                    d_thresh=150, check_origin=False)]
        else:
            objs_sens = objects_ego

        # -- get depth image
        check_origin = False
        if 'CAM' in sens:
            if 'DEPTH' not in sens:
                sens_d = 'DEPTH' + sens
            else:
                sens_d = sens
            try:
                d_img = CDM.get_depthimage(i_frame, sens_d)
            except Exception as e:
                d_img = None
                try:
                    if 'infra' in sens.lower():
                        pc = CDM.get_lidar(i_frame, sens.replace('CAM', 'LIDAR'))  # hack this for now....
                    else:
                        pc = CDM.get_lidar(i_frame, 'LIDAR_TOP')  # hack this for now....
                    check_origin = True
                except Exception as e:
                    logging.warning(e)
                    pc = None
                    print('Could not load depth image...setting occlusion as UNKNOWN')
        elif 'LIDAR' in sens:
            d_img = None
            pc = CDM.get_lidar(i_frame, sens)
        else:
            raise NotImplementedError(sens)

        # -- set occlusion
        for obj in objs_sens:
            if d_img is not None:
                obj.set_occlusion_by_depth(d_img, check_origin=check_origin)
            elif pc is not None:
                obj.set_occlusion_by_lidar(pc, check_origin=check_origin)
            else:
                print('Could not set occlusion!')

        # -- filter to only non-complete, known occlusions
        objs_sens = [obj for obj in objs_sens if obj.occlusion not in [Occlusion.COMPLETE, Occlusion.UNKNOWN]]

    # -- save objects to sensor files
    obj_file = CDM.npc_files['frame'][i_frame]
    CDM.save_objects(i_frame, objs_sens, obj_sens_folder, obj_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()

    main(args)
