# -*- coding: utf-8 -*-
# @Author: spencer@primus
# @Date:   2022-05-30
# @Last Modified by:   spencer@primus
# @Last Modified time: 2022-09-09

import glob
import os
import logging

import numpy as np
from avstack.modules.assignment import gnn_single_frame_assign
from trackeval import Evaluator as TrackEvaluator
from trackeval import _timing
from trackeval.datasets._base_dataset import _BaseDataset as _TrkEvalBaseDataset
from trackeval.metrics import CLEAR, HOTA, VACE, IDEucl, TrackMAP
from trackeval.utils import get_code_path
from trackeval.utils import init_config as init_track_dataset_config

from avapi.evaluation.base import ResultAnalyzer, ResultManager
from avapi.utils import get_indices_in_folder


def get_track_results_from_folder(
    DM,
    result_path,
    idxs=None,
    sensor_eval=None,
    sensor_eval_super=None,
    max_dist=None,
    max_occ=None,
    whitelist_types=None,
    run_tqdm=True,
    multiprocess=True,
):
    if whitelist_types is None:
        whitelist_types = DM.nominal_whitelist_types
    TRA = TrackResultsAnalyzer(
        DM,
        result_path=result_path,
        idxs=idxs,
        sensor_eval=sensor_eval,
        sensor_eval_super=sensor_eval_super,
        max_dist=max_dist,
        max_occ=max_occ,
        whitelist_types=whitelist_types,
        run_tqdm=run_tqdm,
        multiprocess=multiprocess,
    )
    return TRA.results()


def get_track_results_from_multi_folder(DM, result_base, *args, **kwargs):
    res_all = {}
    for subdir in next(os.walk(result_base))[1]:
        result_path = os.path.join(result_base, subdir)
        res_all[subdir] = get_track_results_from_folder(
            DM, result_path, *args, **kwargs
        )
    return res_all


class TrackResultsAnalyzer(ResultAnalyzer):
    @staticmethod
    def _run_per_frame_analysis(
        DM,
        result_path,
        sensor_eval,
        sensor_eval_super,
        max_dist,
        max_occ,
        whitelist_types,
        idx,
    ):
        """Load track status of a single frame"""
        # Get truths
        dist_all = max_dist if sensor_eval_super == "ego" else None
        truths_all = DM.get_objects(
            idx, sensor=sensor_eval_super, max_dist=dist_all, whitelist_types="all"
        )
        truths_whitelist = DM.get_objects(
            idx,
            sensor=sensor_eval,
            max_dist=max_dist,
            max_occ=max_occ,
            whitelist_types=whitelist_types,
        )
        whitelist_ids = [truth.ID for truth in truths_whitelist]
        truths_dontcare = np.asarray(
            [tru for tru in truths_all if tru.ID not in whitelist_ids]
        )

        # Get tracks
        trk_file_path = os.path.join(result_path, "%06i.txt" % idx)
        tracks = DM.get_objects_from_file(
            trk_file_path, whitelist_types=whitelist_types
        )

        # -- 2d analysis

        # -- 3d analysis
        result_3d = ResultManager(
            idx, tracks, truths_whitelist, truths_dontcare, metric="3D_IoU"
        )

        # -- package up
        metrics = {
            "result": result_3d,
            "nTT": result_3d.get_number_of("assigned"),
            "nFT": result_3d.get_number_of("false_positives"),
            "nMT": result_3d.get_number_of("false_negatives"),
            "nT": result_3d.get_number_of("truths"),
        }
        metrics["precision"] = (
            metrics["nTT"] / (metrics["nTT"] + metrics["nFT"])
            if (metrics["nTT"] + metrics["nFT"]) > 0
            else 0
        )
        metrics["recall"] = metrics["nTT"] / metrics["nT"] if metrics["nT"] > 0 else 0
 
        return metrics

    @staticmethod
    def _run_per_seq_analysis(per_frame_metrics, *args, **kwargs):
        tt_per_frame = [m["nTT"] for m in per_frame_metrics.values()]
        ft_per_frame = [m["nFT"] for m in per_frame_metrics.values()]
        mt_per_frame = [m["nMT"] for m in per_frame_metrics.values()]
        t_per_frame = [m["nT"] for m in per_frame_metrics.values()]
        precision = [m["precision"] for m in per_frame_metrics.values()]
        recall = [m["recall"] for m in per_frame_metrics.values()]
        return {
            "tot_TT": sum(tt_per_frame),
            "tot_FT": sum(ft_per_frame),
            "tot_MT": sum(mt_per_frame),
            "tot_T": sum(t_per_frame),
            "mean_precision": np.mean(precision),
            "mean_recall": np.mean(recall),
        }

    @staticmethod
    def _run_expanded_analysis(DM, save_folder,
            sensor_eval, sensor_eval_super, whitelist_types, max_dist):
        """Run HOTA metrics"""
        tracker_name = 'no-name'

        # --- metrics evaluator
        eval_config = TrackEvaluator.get_default_eval_config()
        eval_config["PRINT_CONFIG"] = False
        eval_config["PRINT_RESULTS"] = False
        eval_config["OUTPUT_SUMMARY"] = False
        eval_config["OUTPUT_DETAILED"] = False
        eval_config["TIME_PROGRESS"] = False
        evaluator = TrackEvaluator(eval_config)
        ds = AvstackTrackDataset(
            DM,
            save_folder,
            tracker_name,
            sensor_eval=sensor_eval,
            sensor_eval_super=sensor_eval_super,
            whitelist_types=whitelist_types,
            max_dist=max_dist,
        )
        try:
            eval_results = evaluator.evaluate([ds], [HOTA(), CLEAR(), VACE(), IDEucl()])
        except ValueError:
            logging.warning('Obtained value error during evaluation...skipping')
            mets = {}
        else:
            mets = eval_results[0]["AvstackTrackDataset"][tracker_name][DM.sequence_id]["all_objects"]
            mets = {"{}_{}".format(k1, k2): v2 for k1 in mets for k2, v2 in mets[k1].items()} 
        return mets


# =================================
# From the HOTA paper
# =================================


@staticmethod
def _compute_centroid(obj_3d):
    if isinstance(obj_3d, (list, np.ndarray)):
        centroid = np.array([obj.position.vector for obj in obj_3d])
    else:
        centroid = obj_3d.position.vector
    return centroid

    # box = np.array(box)
    # if len(box.shape) == 1:
    #     centroid = (box[0:2] + box[2:4]) / 2
    # else:
    #     centroid = (box[:, 0:2] + box[:, 2:4]) / 2
    # return np.flip(centroid, axis=1)


IDEucl._compute_centroid = _compute_centroid


class AvstackTrackDataset(_TrkEvalBaseDataset):
    """Dataset class for any AVstack sequence"""

    def __init__(
        self,
        SD,
        trk_folder,
        tracker_name,
        sensor_eval=None,
        sensor_eval_super=None,
        max_dist=None,
        classes="nominal",
        whitelist_types=None,
        ignore_types=None,
        output_folder="metrics",
        output_subfolder="tracking",
        track_has_class=False,
    ):
        """Folders using avstack convention"""
        self.SD = SD
        self.trk_folder = trk_folder
        self.tracker_list = [tracker_name]
        self.sensor_eval = sensor_eval
        self.sensor_eval_super = (
            sensor_eval_super if sensor_eval_super is not None else sensor_eval
        )
        self.max_dist = max_dist
        self.seq_list = [SD.sequence_id]
        self.track_has_class = track_has_class
        if classes == "nominal":
            if track_has_class:
                self.class_list = SD.nominal_whitelist_types
            else:
                self.class_list = ["all_objects"]
            if whitelist_types is not None:
                self.valid_classes = whitelist_types
            else:
                self.valid_classes = SD.nominal_whitelist_types
            if ignore_types is not None:
                self.ignore_list = ignore_types
            else:
                self.ignore_list = SD.nominal_ignore_types
        else:
            raise
            self.class_list
            self.ignore_list = SD.nominal_ignore_types
        self.should_classes_combine = False
        self.use_super_categories = False
        self.output_fol = output_folder
        self.output_sub_fol = output_subfolder
        self.config = init_track_dataset_config(
            None, self.get_default_dataset_config(), self.get_name()
        )

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = get_code_path()
        default_config = {
            "GT_FOLDER": None,  # Location of GT data
            "TRACKERS_FOLDER": None,  # Trackers location
            "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            "TRACKERS_TO_EVAL": None,  # Filenames of trackers to eval (if None, all in folder)
            "CLASSES_TO_EVAL": None,  # Valid: ['car', 'pedestrian']
            "SPLIT_TO_EVAL": None,  # Valid: 'training', 'val'
            "INPUT_AS_ZIP": False,  # Whether tracker input files are zipped
            "PRINT_CONFIG": False,  # Whether to print current config
            "TRACKER_SUB_FOLDER": None,  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            "OUTPUT_SUB_FOLDER": None,  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            "TRACKER_DISPLAY_NAMES": None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            "SEQMAP_FOLDER": None,  # Where seqmaps are found (if None, GT_FOLDER)
            "SEQMAP_FILE": None,  # Directly specify seqmap file (if none use seqmap_folder/split_to_eval.seqmap)
            "SEQ_INFO": None,  # If not None, directly specify sequences to eval and their number of timesteps
            "GT_LOC_FORMAT": None,  # format of gt localization
        }
        return default_config

    def _get_indices_available(self):
        # Check directory
        glob_dir = glob.glob(os.path.join(self.trk_folder, "*.txt"))
        glob_dir = sorted(glob_dir)
        if not os.path.exists(self.trk_folder):
            raise RuntimeError("Cannot find data directory - %s" % self.trk_folder)
        elif len(glob_dir) == 0:
            raise RuntimeError(f"No results to be found in {self.trk_folder}")
        # Get indices by looping
        return get_indices_in_folder(glob_dir, None)

    def _calculate_similarities(
        self, gt_tracks, trk_tracks, cost_thresh=0.02, metric="center_dist", radius=4
    ):
        """Computes matrix of similarities with gt as rows, trks as cols"""
        A = np.zeros((len(gt_tracks), len(trk_tracks)))
        for i, gt in enumerate(gt_tracks):
            for j, trk in enumerate(trk_tracks):
                if metric == "IoU":
                    IoU = gt.box.IoU(trk.box)  # higher is better
                    if IoU <= cost_thresh:
                        IoU = 0
                    cost = IoU
                elif metric == "center_dist":
                    if trk.box.origin != gt.box.origin:
                        trk.box.change_origin(gt.box.origin)
                    dist = gt.box.t.distance(trk.box.t)  # lower is better
                    if dist >= radius:
                        cost = 0
                    else:
                        cost = radius - dist  # higher is better
                else:
                    raise NotImplementedError(metric)

                A[i, j] = cost
        return A

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load the results (gt or tracker) in AVstack's format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets]: list (for each timestep) of lists of detections.
        [gt_ignore_region]: list (for each timestep) of masks for the ignore regions

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        if is_gt:
            raw_data = {"gt_ids": [], "gt_classes": [], "gt_dets": [], "gt_ignores": []}
            for idx in self._get_indices_available():
                # -- get objects
                dist_all = self.max_dist if self.sensor_eval_super == "ego" else None
                truths_all = self.SD.get_objects(
                    idx,
                    sensor=self.sensor_eval_super,
                    max_dist=dist_all,
                    whitelist_types="all",
                )
                truths_whitelist = self.SD.get_objects(
                    idx,
                    sensor=self.sensor_eval,
                    max_dist=self.max_dist,
                    whitelist_types=self.valid_classes,
                    ignore_types=self.ignore_list,
                )
                whitelist_ids = [truth.ID for truth in truths_whitelist]
                truths_dontcare = np.asarray(
                    [tru for tru in truths_all if tru.ID not in whitelist_ids]
                )

                # -- store results
                raw_data["gt_ids"].append(
                    np.array([truth.ID for truth in truths_whitelist])
                )
                if self.track_has_class:
                    raw_data["gt_classes"].append(
                        np.array([truth.obj_type for truth in truths_whitelist])
                    )
                else:
                    raw_data["gt_classes"].append(
                        np.array(["all_objects" for truth in truths_whitelist])
                    )
                raw_data["gt_dets"].append(truths_whitelist)
                raw_data["gt_ignores"].append(truths_dontcare)
        else:
            raw_data = {"tracker_ids": [], "tracker_classes": [], "tracker_dets": []}
            for idx in self._get_indices_available():
                trk_file_path = os.path.join(self.trk_folder, "%06i.txt" % idx)
                tracks = self.SD.get_objects_from_file(
                    trk_file_path,
                    whitelist_types=self.valid_classes,
                    max_dist=self.max_dist,
                )
                # -- store results
                raw_data["tracker_ids"].append(np.array([trk.ID for trk in tracks]))
                if self.track_has_class:
                    raw_data["tracker_classes"].append(
                        np.array([trk.obj_type for trk in tracks])
                    )
                else:
                    raw_data["tracker_classes"].append(
                        np.array(["all_objects" for trk in tracks])
                    )
                raw_data["tracker_dets"].append(tracks)
        raw_data["num_timesteps"] = len(self._get_indices_available())
        raw_data["seq"] = self.SD.sequence_id

        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, obj_type):
        """Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - obj_type is the class (object class) to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detection masks.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        AVSTACK MOTS:
            In AVSTACK MOTS, the 3 preproc steps are as follow:
                1) Evaluate each class separately
                2) Remove ignore types and tracks matched with ignore types
                3) Ignore regions are used to remove unmatched detections (at least 50% overlap with ignore region).
        """
        data = {
            "gt_ids": [],
            "tracker_ids": [],
            "gt_dets": [],
            "tracker_dets": [],
            "similarity_scores": [],
        }

        # loop per timestep
        for t in range(raw_data["num_timesteps"]):
            # 1) filter out the truths and detections by class
            if len(raw_data["gt_classes"][t]) > 0:
                gt_class_mask = raw_data["gt_classes"][t] == obj_type
            else:
                gt_class_mask = np.zeros((0,), dtype=np.bool)
            gt_ids = raw_data["gt_ids"][t][gt_class_mask]
            gt_trks = raw_data["gt_dets"][t][gt_class_mask]
            if len(raw_data["tracker_classes"][t]) > 0:
                trk_class_mask = raw_data["tracker_classes"][t] == obj_type
            else:
                trk_class_mask = np.zeros((0,), dtype=np.bool)
            trk_ids = raw_data["tracker_ids"][t][trk_class_mask]
            trk_trks = raw_data["tracker_dets"][t][trk_class_mask]
            sim_scores = raw_data["similarity_scores"][t][gt_class_mask, :][
                :, trk_class_mask
            ]

            # 2) perform assignment, remove anything near "ignores"
            ignore_cols = []
            if len(raw_data["gt_ignores"][t]) > 0:
                sim_scores_ignores = self._calculate_similarities(
                    raw_data["gt_ignores"][t], trk_trks, metric="center_dist"
                )
                assign_sol = gnn_single_frame_assign(
                    -sim_scores_ignores,
                    algorithm="JVC",
                    cost_threshold=-0.02,
                    all_assigned=False,
                )
                for r, c in assign_sol.assignment_tuples:
                    ignore_cols.append(c)  # column is the track
            # sanity check:
            if (len(ignore_cols) > 0) and (
                len(ignore_cols) > np.sum(sim_scores_ignores > 0)
            ):
                raise RuntimeError("Something is up with ignores")
            trk_ids = [t_id for i, t_id in enumerate(trk_ids) if i not in ignore_cols]
            trk_trks = [trk for i, trk in enumerate(trk_trks) if i not in ignore_cols]
            sim_scores = np.delete(sim_scores, ignore_cols, axis=1)

            # final) package up
            data["gt_ids"].append(gt_ids)
            data["gt_dets"].append(gt_trks)
            data["tracker_ids"].append(trk_ids)
            data["tracker_dets"].append(trk_trks)
            data["similarity_scores"].append(sim_scores)

        # final touches - relabel
        unique_gt_ids = np.array(
            list({gt_id for gts_id in data["gt_ids"] for gt_id in gts_id})
        )
        unique_trk_ids = np.array(
            list({trk_id for trks_id in data["tracker_ids"] for trk_id in trks_id})
        )
        gt_id_map = {gt_id: i for i, gt_id in enumerate(unique_gt_ids)}
        trk_id_map = {trk_id: i for i, trk_id in enumerate(unique_trk_ids)}
        num_gt_dets = num_tracker_dets = 0
        for t in range(raw_data["num_timesteps"]):
            data["gt_ids"][t] = np.array(
                [gt_id_map[gt_id] for gt_id in data["gt_ids"][t]], dtype=np.int
            )
            data["tracker_ids"][t] = np.array(
                [trk_id_map[trk_id] for trk_id in data["tracker_ids"][t]], dtype=np.int
            )
            num_gt_dets += len(data["gt_dets"])
            num_tracker_dets += len(data["tracker_dets"])

        # final touches - overview stats
        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_trk_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]
        data["seq"] = raw_data["seq"]
        data["cls"] = obj_type

        return data
