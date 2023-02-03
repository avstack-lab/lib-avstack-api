# -*- coding: utf-8 -*-
# @Author: spencer@primus
# @Date:   2022-06-28
# @Last Modified by:   spencer@primus
# @Last Modified time: 2022-09-09

import os

import numpy as np
from avstack.geometry import NominalTransform as nom_trans
from avstack.geometry import bbox
from avstack.modules.perception.detections import BoxDetection, get_detections_from_file

from avapi.evaluation.base import ResultAnalyzer, ResultManager


# from avapi.evaluation.base import ResultManager


def get_percep_results_from_folder(
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
    PRA = PercepResultsAnalyzer(
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
    return PRA.results()


def get_percep_results_from_multi_folder(DM, result_base, *args, **kwargs):
    res_all = {}
    for subdir in next(os.walk(result_base))[1]:
        result_path = os.path.join(result_base, subdir)
        res_all[subdir] = get_percep_results_from_folder(
            DM, result_path, *args, **kwargs
        )
    return res_all


class PercepResultsAnalyzer(ResultAnalyzer):
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
        """Load percep status of a single frame"""
        # Get truths
        # import ipdb; ipdb.set_trace()
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

        # Get detections
        det_file_path = os.path.join(result_path, "%06i.txt" % idx)
        detections = get_detections_from_file(det_file_path)
        dets = [det for det in detections if isinstance(det, (BoxDetection,))]
        metric = (
            "3D_IoU"
            if (len(dets) == 0) or (isinstance(dets[0].box, bbox.Box3D))
            else "2D_IoU"
        )
        result = ResultManager(
            idx, dets, truths_whitelist, truths_dontcare, metric=metric
        )

        # -- package up
        metrics = {
            "result": result,
            "nTP": result.get_number_of("assigned"),
            "nFP": result.get_number_of("false_positives"),
            "nFN": result.get_number_of("false_negatives"),
            "nT": result.get_number_of("truths"),
        }
        metrics["precision"] = (
            metrics["nTP"] / (metrics["nTP"] + metrics["nFP"])
            if (metrics["nTP"] + metrics["nFP"]) > 0
            else 0
        )
        metrics["recall"] = metrics["nTP"] / metrics["nT"] if metrics["nT"] > 0 else 0
        return metrics

    @staticmethod
    def _run_per_seq_analysis(per_frame_metrics, *args, **kwargs):
        tp_per_frame = [m["nTP"] for m in per_frame_metrics.values()]
        fp_per_frame = [m["nFP"] for m in per_frame_metrics.values()]
        fn_per_frame = [m["nFN"] for m in per_frame_metrics.values()]
        t_per_frame = [m["nT"] for m in per_frame_metrics.values()]
        precision = [m["precision"] for m in per_frame_metrics.values()]
        recall = [m["recall"] for m in per_frame_metrics.values()]
        return {
            "tot_TP": sum(tp_per_frame),
            "tot_FP": sum(fp_per_frame),
            "tot_FN": sum(fn_per_frame),
            "tot_T": sum(t_per_frame),
            "mean_precision": np.mean(precision),
            "mean_recall": np.mean(recall),
        }

    @staticmethod
    def _run_expanded_analysis(*args, **kwargs):
        return None