# -*- coding: utf-8 -*-
# @Author: spencer@primus
# @Date:   2022-05-30
# @Last Modified by:   Spencer H
# @Last Modified time: 2022-05-30

import os, logging
import numpy as np
from copy import copy, deepcopy
import avstack
from avstack.modules import tracking
from avstack.modules import prediction
from avstack.modules.perception.detections import BoxDetection
from avstack.datastructs import DataContainer, DataManager

import avapi
from avapi.kitti import KittiObjectDataset as KOD


KITTI_data_dir = os.path.join(os.getcwd(), 'data/KITTI/object')
if os.path.exists(os.path.join(KITTI_data_dir, 'training')):
    KDM_train = KOD(KITTI_data_dir, 'training')
else:
    KDM_train = None
    msg_obj = "Cannot run test - KITTI object data not downloaded"
new_folder = os.path.join(KITTI_data_dir, 'tracking_test')
new_label_folder = os.path.join(new_folder, 'label_2')
new_calib_folder = os.path.join(new_folder, 'calib')
os.makedirs(new_folder, exist_ok=True)

np.random.seed(5)

name_3d = 'detector-3d'
name_2d = 'detector-2d'
idx_frame = 100


def make_kitti_tracking_data(KDM, idx_frame, dt=0.1, n_frames=10):
    truths = KDM.get_objects(idx_frame)
    calib_camera = KDM.get_calibration(idx_frame, 'image-2')
    det_manager = DataManager(max_size=np.inf)
    v_cam = 5*np.random.randn(len(truths))
    t = 0
    for i in range(n_frames):
        labs = deepcopy(truths)
        dets_class = []
        labs_new = []
        for lab, v in zip(labs, v_cam):
            lab.box3d.t[2] += t * v  # camera coordinates with z forward
            lab.box2d = lab.box3d.project_to_2d_bbox(calib=calib_camera)
            det = BoxDetection(name_3d, lab.box3d, lab.obj_type)
            dets_class.append(det)
            labs_new.append(lab)
        KDM.save_calibration(i, calib_camera, new_calib_folder, sensor_ID=2)
        KDM.save_objects(i, labs_new, new_label_folder)
        detections = DataContainer(i, t, dets_class, source_identifier=name_3d)
        det_manager.push(detections)
        t += dt
    imset_path = KOD._get_imset_path_from_data_path(new_folder)
    KOD._write_imset(imset_path, list(range(n_frames)))
    return det_manager


def run_tracker(tracker, det_manager, predictor=None):
    frame = 0
    while not det_manager.empty():
        dets = det_manager.pop(name_3d)
        tracks = tracker(frame, dets)
        if predictor is not None:
            predictions = predictor(frame, tracks)
        frame += 1
    return tracks


def test_eval_tracks():
    save_folder = os.path.join(os.getcwd(), 'tmp', 'results')

    if KDM_train is not None:
        # track
        detections = make_kitti_tracking_data(KDM_train, 100, n_frames=10)
        tracker = tracking.tracker3d.BasicBoxTracker(framerate=10, save_output=True, save_folder=save_folder)
        tracks = run_tracker(tracker, detections)

        # evaluate
        KDM_new = KOD(KITTI_data_dir, new_folder)
        res_folder = os.path.join(save_folder, 'tracking')
        res_frame, res_seq = avapi.evaluation.get_track_results_from_folder(KDM_new, res_folder, multiprocess=False)
        assert len(res_frame) == len(KDM_new.frames)
    else:
        logging.warning(msg_obj)
        print(msg_obj)


def test_eval_preds():
    save_folder = os.path.join(os.getcwd(), 'tmp', 'results')

    if KDM_train is not None:
        # track
        detections = make_kitti_tracking_data(KDM_train, 100, n_frames=10)
        tracker = tracking.tracker3d.BasicBoxTracker(framerate=10, save_output=True, save_folder=save_folder)
        predictor = prediction.KinematicPrediction(dt_pred=0.1, t_pred_forward=3, save_output=True, save_folder=save_folder)
        tracks = run_tracker(tracker, detections, predictor)

        # evaluate predictions
        KDM_new = KOD(KITTI_data_dir, new_folder)
        res_folder = os.path.join(save_folder, 'prediction')
        pred_results = avapi.evaluation.get_predict_results_from_folder(KDM_new, res_folder, multiprocess=False)
        assert len(pred_results) > 0
    else:
        logging.warning(msg_obj)
        print(msg_obj)
