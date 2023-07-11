import bisect
import heapq
import os

import numpy as np

from avapi.evaluation.base import ResultAnalyzer, ResultManager


def get_predict_results_from_folder(
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
    PRA = PredictResultsAnalyzer(
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


def get_predict_results_from_multi_folder(DM, result_base, *args, **kwargs):
    res_all = {}
    for subdir in next(os.walk(result_base))[1]:
        result_path = os.path.join(result_base, subdir)
        res_all[subdir] = get_predict_results_from_folder(
            DM, result_path, *args, **kwargs
        )
    return res_all


class PredictResultsAnalyzer(ResultAnalyzer):
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
        max_objects=30,
    ):
        """Load prediction status of a single frame

        NOTE: this could be made faster by pre-loading and sharing "look-ahead" true objects

        idx: frame of the prediction made

        Intermediate:
        :predictions a dictionary of {obj_ID:{time:{results...}}]}
        """
        # Get timestamp to frame mappings
        t_to_frame = {}
        try:
            t_to_frame = {v: k for k, v in DM.ego_frame_to_ts.items()}
        except AttributeError as e:
            dt = 1.0 / DM.framerate
            t_to_frame = {dt * frame: frame for frame in DM.frames}
        ts_available = list(t_to_frame.keys())

        # Get predictions
        pred_file_path = os.path.join(result_path, "%06i.txt" % idx)
        predictions_raw = DM.get_objects_from_file(
            pred_file_path, whitelist_types=whitelist_types
        )

        # Organize predictions into object ID:timestamp:object
        predictions = {}
        time_to_frame_map = {}
        for pred in predictions_raw:
            if pred.ID not in predictions:
                if len(predictions) >= max_objects:
                    continue
                predictions[pred.ID] = {}
            if pred.t > (ts_available[-1] + 1e-6):
                continue
            predictions[pred.ID][pred.t] = {
                "prediction": pred,
                "truth": None,
                "truth_ID": None,
                "displacement": None,
            }
            idx_closest = bisect.bisect_left(ts_available, pred.t)
            time_to_frame_map[pred.t] = t_to_frame[ts_available[idx_closest]]

        # Get truth objects for corresponding frames predicted to
        frames_all = np.unique(list(time_to_frame_map.values()))
        truths_all = {}
        dist_all = max_dist if sensor_eval_super == "ego" else None
        for frame in frames_all:
            truths_all[frame] = DM.get_objects(
                frame,
                sensor=sensor_eval_super,
                max_dist=dist_all,
                whitelist_types="all",
            )

        # Get assignments (1 to 1)
        for pred_ID in predictions:
            # -- assign at first prediction to get truth ID
            t_pred = list(predictions[pred_ID].keys())
            if len(t_pred) == 0:
                continue
            else:
                t_pred = t_pred[0]
            pred = [pred.box3d for pred in [predictions[pred_ID][t_pred]["prediction"]]]
            frame = time_to_frame_map[t_pred]
            truths = [tru.box3d for tru in truths_all[frame]]
            result_3d = ResultManager(frame, pred, truths, [], metric="3D_IoU")
            asgn_tups = result_3d.assignment.assignment_tuples
            if len(asgn_tups) == 1:
                truth_ID_assign = truths[asgn_tups[0][1]].ID

                # -- run over predicted timesteps
                for t_pred in predictions[pred_ID]:
                    pred = [predictions[pred_ID][t_pred]["prediction"]]
                    frame = time_to_frame_map[t_pred]
                    truths = truths_all[frame]
                    truth_this = [
                        truth for truth in truths if truth.ID == truth_ID_assign
                    ]
                    if len(truth_this) == 0:
                        continue
                    elif len(truth_this) == 1:
                        predictions[pred_ID][t_pred]["truth"] = truth_this[0]
                        predictions[pred_ID][t_pred]["truth_ID"] = truth_ID_assign
                        predictions[pred_ID][t_pred]["displacement"] = predictions[
                            pred_ID
                        ][t_pred]["prediction"].position.distance(
                            truth_this[0].position
                        )
                    else:
                        raise RuntimeError("Not supposed to find more than 1 truth")

        # Calculate per-prediction metrics
        metrics = calculate_prediction_metrics(predictions)

        return metrics

    @staticmethod
    def _run_per_seq_analysis(per_frame_metrics, *args, **kwargs):
        """
        Calculate aggregate sequence metrics using the information from the
        per-frame runs
        """
        n_with_truth = sum(
            [
                m["n_with_truth"]
                for m in per_frame_metrics.values()
                if m["n_with_truth"] > 0
            ]
        )
        n_tot = sum(
            [m["n_objects"] for m in per_frame_metrics.values() if m["n_objects"] > 0]
        )
        weighted_ADE = [
            m["ADE"] * m["n_with_truth"]
            for m in per_frame_metrics.values()
            if m["ADE"] is not None
        ]
        weighted_FDE = [
            m["FDE"] * m["n_with_truth"]
            for m in per_frame_metrics.values()
            if m["FDE"] is not None
        ]

        return {
            "agg_ADE": sum(weighted_ADE) / n_with_truth if n_with_truth > 0 else None,
            "agg_FDE": sum(weighted_FDE) / n_with_truth if n_with_truth > 0 else None,
            "std_ADE": np.std(
                [m["ADE"] for m in per_frame_metrics.values() if m["ADE"] is not None]
            ),
            "std_FDE": np.std(
                [m["FDE"] for m in per_frame_metrics.values() if m["FDE"] is not None]
            ),
            "n_with_truth": n_with_truth,
            "n_objects": n_tot,
        }

    @staticmethod
    def _run_expanded_analysis(*args, **kwargs):
        return None


# =========================================
# UTILITIES
# =========================================


def calculate_prediction_metrics(predictions):
    """
    Calculate ADE and FDE metrics on a per-frame basis
    :predictions -- a dictionary as defined above
    """
    return {
        "ADE": calculate_ADE(predictions),
        "FDE": calculate_FDE(predictions),
        "n_objects": len(predictions),
        "n_with_truth": calculate_n_with_truth(predictions),
    }


def calculate_n_with_truth(predictions):
    has_truth = []
    for obj_ID in predictions:
        for t_pred in predictions[obj_ID]:
            if predictions[obj_ID][t_pred] is not None:
                has_truth.append(True)
                break
        else:
            has_truth.append(False)
    return sum(has_truth)


def calculate_ADE(predictions):
    """
    ADE is defined as the MSE of all displacement errors from a single prediction set

    Here, we use the median to remove outliers...

    see: https://jaimefernandezdcu.wordpress.com/2019/02/07/error-metrics-for-trajectory-prediction-accuracy/
    """
    disp_errors = [
        pred["displacement"]
        for preds in predictions.values()
        for pred in preds.values()
        if pred["displacement"] is not None
    ]
    if len(disp_errors) > 0:
        return np.median(disp_errors)
    else:
        return None


def calculate_FDE(predictions):
    """
    FDE is defined as the MSE of the final errors from a single prediction set

    Here, we use the median to remove outliers...

    We also cut short in case track is lost

    see: https://jaimefernandezdcu.wordpress.com/2019/02/07/error-metrics-for-trajectory-prediction-accuracy/
    """
    final_disps = []
    final_disps = [
        [
            pred["displacement"]
            for pred in preds.values()
            if pred["displacement"] is not None
        ]
        for preds in predictions.values()
    ]
    final_disps = [fd[-1] for fd in final_disps if len(fd) > 0]
    if len(final_disps) > 0:
        return np.median(final_disps)
    else:
        return None


# def calculate_ADE(test_label, predicted_output, test_num, predicting_frame_num, show_num):
#     """
#     According to https://github.com/xuehaouwa/SS-LSTM
#
#     :truth -- true object
#     :predictino -- predicted object
#
#     """
#     total_ADE = np.zeros((test_num, 1))
#     for i in range(test_num):
#         predicted_result_temp = predicted_output[i]
#         label_temp = test_label[i]
#         ADE_temp = 0.0
#         for j in range(predicting_frame_num):
#             ADE_temp += distance.euclidean(predicted_result_temp[j], label_temp[j])
#         ADE_temp = ADE_temp / predicting_frame_num
#         total_ADE[i] = ADE_temp
#
#     show_ADE = heapq.nsmallest(show_num, total_ADE)
#
#     show_ADE = np.reshape(show_ADE, [show_num, 1])
#
#     return np.average(show_ADE)


# def calculate_FDE(test_label, predicted_output, test_num, show_num):
#     """
#     According to https://github.com/xuehaouwa/SS-LSTM
#     """
#     total_FDE = np.zeros((test_num, 1))
#     for i in range(test_num):
#         predicted_result_temp = predicted_output[i]
#         label_temp = test_label[i]
#         total_FDE[i] = distance.euclidean(predicted_result_temp[-1], label_temp[-1])
#
#     show_FDE = heapq.nsmallest(show_num, total_FDE)
#
#     show_FDE = np.reshape(show_FDE, [show_num, 1])
#
#     return np.average(show_FDE)
