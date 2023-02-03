# @Author: Spencer Hallyburton <spencer>
# @Date:   2022-04-18
# @Filename: base.py
# @Last modified by:   spencer
# @Last modified time: 2022-04-18

import glob
import os
from copy import copy, deepcopy
from functools import partial
from multiprocessing import Pool

import numpy as np
from avstack.datastructs import OneEdgeBipartiteGraph
from avstack.maskfilters import box_in_fov
from avstack.modules.assignment import gnn_single_frame_assign
from tqdm import tqdm

from avapi.utils import get_indices_in_folder


def color_from_object_type(det_type, no_white=False, no_black=False):
    if det_type == "detection":
        cstring = "green" if no_black else "black"
    elif det_type == "truth":
        cstring = "cyan" if no_white else "white"
    elif det_type == "false_negative":
        cstring = "yellow"
    elif det_type == "false_positive":
        cstring = "red"
    elif det_type == "true_positive":
        cstring = "blue"
    elif det_type == "dontcare":
        cstring = "brown"
    else:
        raise NotImplementedError(f"{det_type} not available for color")
    return parse_color_string(cstring)


def parse_color_string(cstring):
    if cstring == "white":
        lcolor = (255, 255, 255)
    elif cstring == "green":
        lcolor = (0, 255, 0)
    elif cstring == "red":
        lcolor = (255, 0, 0)
    elif cstring == "blue":
        lcolor = (0, 0, 255)
    elif cstring == "cyan":
        lcolor = (0, 255, 255)
    elif cstring == "lightblue":
        lcolor = (51, 255, 255)
    elif cstring == "black":
        lcolor = (0, 0, 0)
    elif cstring == "yellow":
        lcolor = (236, 213, 64)
        # lcolor = (255, 255, 0)
    elif cstring == "brown":
        lcolor = (165, 42, 42)
    else:
        raise ValueError(f"Unknown color type {cstring}")
    return lcolor


class ResultAnalyzer:
    def __init__(
        self,
        DM,
        result_path,
        idxs=None,
        sensor_eval=None,
        max_dist=None,
        sensor_eval_super=None,
        max_occ=None,
        whitelist_types=["Car", "Pedestrian", "Cyclist"],
        run_tqdm=True,
        multiprocess=True,
    ):
        self.DM = DM
        self.result_path = result_path
        self.idxs = idxs
        self.sensor_eval = sensor_eval
        self.sensor_eval_super = (
            sensor_eval_super if sensor_eval_super is not None else sensor_eval
        )
        self.max_dist = max_dist
        self.max_occ = max_occ
        self.whitelist_types = whitelist_types
        self.run_tqdm = run_tqdm
        self.multiprocess = multiprocess

    def results(self):
        res_frame, res_seq, res_exp = self._results_from_folder()
        return res_frame, res_seq, res_exp


    def _results_from_folder(self):
        # Check directory
        glob_dir = glob.glob(os.path.join(self.result_path, "*.txt"))
        glob_dir = sorted(glob_dir)
        if not os.path.exists(self.result_path):
            raise RuntimeError("Cannot find data directory - %s" % self.result_path)
        elif len(glob_dir) == 0:
            raise RuntimeError(f"No results to be found in {self.result_path}")

        # Get indices by looping
        self.idxs_available = get_indices_in_folder(glob_dir, self.idxs)

        # -- run per-frame analysis
        part_func = partial(
            self._run_per_frame_analysis,
            self.DM,
            self.result_path,
            self.sensor_eval,
            self.sensor_eval_super,
            self.max_dist,
            self.max_occ,
            self.whitelist_types,
        )
        res_frame = []
        if self.multiprocess:
            with Pool(8) as p:
                res_frame = list(
                    tqdm(
                        p.imap(part_func, self.idxs_available),
                        total=len(self.idxs_available),
                    )
                )
        else:
            for idx in tqdm(self.idxs_available):
                res_frame.append(part_func(idx))

        res_frame = {idx: res for idx, res in zip(self.idxs_available, res_frame)}

        # -- run per-sequence analysis
        res_seq = self._run_per_seq_analysis(res_frame)

        # -- run expanded analysis
        res_exp = self._run_expanded_analysis(self.DM, self.result_path, self.sensor_eval, self.sensor_eval_super, self.whitelist_types, self.max_dist)

        return res_frame, res_seq, res_exp

    @staticmethod
    def _run_per_frame_analysis():
        raise NotImplementedError

    @staticmethod
    def _run_per_seq_analysis():
        raise NotImplementedError

    @staticmethod
    def _run_expanded_analysis():
        raise NotImplementedError


class ResultManager:
    def __init__(
        self,
        idx,
        detections,
        truths,
        truths_dontcare=[],
        metric="3D_IoU",
        radius=None,
        no_white=False,
        no_black=False,
    ):
        """ """
        self.idx = idx
        self.metric = metric
        self.radius = radius
        self.detections = detections
        self.truths = truths
        self.truths_dontcare = truths_dontcare
        self.truths_all = np.concatenate((truths, np.asarray(truths_dontcare)), axis=0)
        self.colors = {"detections": [], "truths": []}
        self.no_white = no_white
        self.no_black = no_black
        self.run_assignment()

    @property
    def tracks(self):
        return self.detections

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Result Manager\n----{} detections, {} truths\n----{} detections are dontcares\n----{} Assignments, {} FPs, {} FNs".format(
            len(self.detections),
            len(self.truths),
            len(self.result["det_dontcare"]),
            len(self.result["assigned"]),
            len(self.result["false_positives"]),
            len(self.result["false_negatives"]),
        )

    def get_confusion(self):
        """
        [[nTP, nFN]
         [nFP, nTN]]
        """
        return self.confusion

    def run_assignment(self):
        # Initialize variables
        self.result = {
            "false_positives": [],
            "false_negatives": [],
            "assigned": [],
            "det_dontcare": [],
        }
        self.confusion = np.zeros((2, 2))

        # Put detections into the coordinate frame of the truths
        assert all(
            [tru.origin == self.truths[0].origin for tru in self.truths]
        ), "All truth origins must be the same for now"

        # Run assignment
        dets = deepcopy(self.detections)
        assignment, A = associate_detections_truths(
            dets, self.truths_all, self.metric, self.radius
        )
        self.A = A

        # Remove assignments and FNs in dontcare
        idx_FP = deepcopy(assignment.unassigned_rows)
        idx_tru_dontcare = list(
            range(len(self.truths), len(self.truths) + len(self.truths_dontcare))
        )
        idx_det_dontcare = [
            r
            for r, c in assignment.iterate_over("rows").items()
            if c in idx_tru_dontcare
        ]
        assigns = {
            r: c
            for r, c in assignment.iterate_over("rows").items()
            if list(c.keys())[0] not in idx_tru_dontcare
        }
        assignment = OneEdgeBipartiteGraph(
            assigns,
            assignment.nrow - len(idx_det_dontcare),
            assignment.ncol - len(idx_tru_dontcare),
            assignment.cost,
        )
        idx_FN = deepcopy(assignment.unassigned_cols)

        # Make confusion matrix
        self.confusion = np.array([[len(assignment), len(idx_FN)], [len(idx_FP), 0]])

        # Store map from index to what it is
        self.result["false_positives"] = idx_FP
        self.result["false_negatives"] = idx_FN
        self.result["det_dontcare"] = idx_det_dontcare
        self.result["assigned"] = assignment._row_to_col
        self.assignment = assignment
        iou_assign = []

        # Get IoU for each assignment
        for ia, ib in self.result["assigned"].items():
            try:
                ib = list(ib.keys())[0]
                iou_assign.append(dets[ia].box3d.IoU(self.truths[ib].box3d))
            except AttributeError as e:
                self.result["assigned_iou"] = []
            else:
                self.result["assigned_iou"] = np.asarray(iou_assign)

        # Assign colors
        for i in range(len(self.detections)):
            if i in idx_FP:
                self.colors["detections"].append(
                    color_from_object_type(
                        "false_positive", self.no_white, self.no_black
                    )
                )
            elif i in idx_det_dontcare:
                self.colors["detections"].append(
                    color_from_object_type("dontcare", self.no_white, self.no_black)
                )
            else:
                self.colors["detections"].append(
                    color_from_object_type(
                        "true_positive", self.no_white, self.no_black
                    )
                )
        for i in range(len(self.truths)):
            if i in idx_FN:
                self.colors["truths"].append(
                    color_from_object_type(
                        "false_negative", self.no_white, self.no_black
                    )
                )
            elif i in idx_tru_dontcare:
                self.colors["truths"].append(
                    color_from_object_type("dontcare", self.no_white, self.no_black)
                )
            else:
                self.colors["truths"].append(
                    color_from_object_type("truth", self.no_white, self.no_black)
                )

    def get_assignment_iou(self, idx_det=None, idx_tru=None):
        dk = list(dict(self.result["assigned"]).keys())
        dv = list(dict(self.result["assigned"]).values())

        if (idx_det is None) and (idx_tru is None):
            iou = self.result["assigned_iou"]
        elif (idx_det is not None) and (idx_tru is None):
            if idx_det in dk:
                iou = self.result["assigned_iou"][dk.index(idx_det)]
            else:
                iou = None
        elif (idx_det is None) and (idx_tru is not None):
            if idx_tru in dv:
                iou = self.result["assigned_iou"][dv.index(idx_tru)]
            else:
                iou = None
        else:
            found = False
            for iou, (i, j) in zip(
                self.result["assigned_iou"], self.result["assigned"]
            ):
                if (i == idx_det) and (j == idx_tru):
                    found = True
                    break
            if not found:
                iou = None
        return iou

    def get_number_of(self, field):
        if field in ["false_positives", "false_negatives", "det_dontcare"]:
            return len(self.result[field])
        elif field == "detections":
            return len(self.detections)
        elif field == "truths":
            return len(self.truths)
        elif field in ["assigned", "assign", "assignments"]:
            return len(self.result["assigned"])
        else:
            raise IOError("Cannot understand %s field type" % field)

    def get_indices_of(self, field):
        if field in ["false_positives", "false_negatives", "assigned", "det_dontcare"]:
            return self.result[field]
        elif field in ["detections", "truths"]:
            return list(range(self.get_number_of(field)))
        else:
            raise IOError("Cannot understand %s field type" % field)

    def get_objects_of(self, field):
        if field == "false_positives":
            return [
                det
                for i, det in enumerate(self.detections)
                if i in self.result["false_positives"]
            ]
        elif field == "false_negatives":
            return [
                tru
                for i, tru in enumerate(self.truths)
                if i in self.result["false_negatives"]
            ]
        elif field == "assigned":
            return [
                (self.detections[i], self.truths[list(j.keys())[0]])
                for i, j in self.result["assigned"].items()
            ]
        elif field == "truths":
            return self.truths
        elif field == "detections":
            return self.detections
        else:
            raise IOError("Cannot understand %s field type" % field)

    def remove_truth(self, label=None, idx=None):
        assert (label is None) or (idx is None)
        if label is not None:
            if self.truths is list:
                self.truths.remove(label)
            else:
                self.truths = np.delete(self.truths, label)
        elif idx is not None:
            del self.truths[idx]
        else:
            raise IOError("Must pass in something")
        self.run_assignment()

    def remove_detection(self, label=None, idx=None):
        """Remove detection and rerun assignment"""
        assert (label is None) or (idx is None)
        if label is not None:
            if self.detections is list:
                self.detections.remove(label)
            else:
                self.detections = np.delete(self.detections, label)
        elif idx is not None:
            del self.detections[idx]
        else:
            raise IOError("Must pass in something")
        self.run_assignment()

    def visualize(self, DM=None, image=None, lidar=None, calib=None, projection="bev"):
        # Get colors
        labels_all = np.hstack([self.detections, self.truths])
        colors_all = self.colors["detections"] + self.colors["truths"]

        if (DM is not None) and (calib is None):
            calib = DM.get_calibration(self.idx)

        # ----- Show result in some projection
        if not isinstance(projection, list):
            projection = [projection]
        for proj in projection:
            if proj == "bev":
                """Takes 3D result and projects into BEV"""
                assert ((lidar is not None) and (calib is not None)) or (DM is not None)
                if (DM is not None) and (lidar is None):
                    lidar = DM.get_lidar(self.idx)
                visualize.show_lidar_bev(
                    lidar, calib=calib, labels=labels_all, label_colors=colors_all
                )
            elif proj == "2d":
                """Assume front-view camera result"""
                assert ((image is not None) and (calib is not None)) or (DM is not None)
                if (DM is not None) and (image is None):
                    image = DM.get_image(self.idx)
                visualize.show_image_with_boxes(
                    image,
                    calib=calib,
                    labels=labels_all,
                    label_colors=colors_all,
                    show3d=False,
                    inline=True,
                )
            elif proj == "fv":
                """Takes 3D result and projects into FV"""
                raise NotImplementedError
            else:
                raise NotImplementedError


def associate_detections_truths(detections, truths, metric="3D_IoU", radius=None):
    """
    Determine associations, false positives, and false negatives

    Figure out which detections are associated to which truths,
    which detections are false alarms,
    and which truths are false negatives

    INPUTS:
    detections --> a list of label classes for detection
    truths     --> a list of label classes for truths

    OUTPUT:
    assignments --> list of tuples of assignments (detection --> truth)
    idx_FP      --> indices in the detection list of false positives
    idx_FN      --> indices in the truth list of false negatives
    """

    tol = 1e-8

    # Handle case of none
    no_dets = (detections is None) or (len(detections) == 0)
    no_truths = (truths is None) or (len(truths) == 0)

    if no_dets or no_truths:
        nr = 0 if detections is None else len(detections)
        nc = 0 if truths is None else len(truths)
        A = np.zeros((nr, nc))
    else:
        A = _build_A_matrix(detections, truths, metric, radius)

    assignment = gnn_single_frame_assign(A, algorithm="JVC", all_assigned=False)
    return assignment, A


def _build_A_matrix(detections, truths, metric, radius):
    if radius is not None:
        radius2 = radius**2
    else:
        radius2 = None

    # Build assignment matrix
    A = np.zeros((len(detections), len(truths)))
    # Add IoU's to the assignment matrix
    for i, det in enumerate(detections):
        for j, tru in enumerate(truths):
            A[i, j] = _get_assignment_cost(det, tru, metric, radius2)
    # import ipdb; ipdb.set_trace()
    return A


def _get_center(obj):
    try:
        center = obj.box3d.center
    except AttributeError as e:
        npos = sum(["position" in s for s in obj.filter.state_names])
        center = np.squeeze(obj.filter.x_vector[:npos])
    return center


def _get_assignment_cost(det, tru, metric, radius2):
    """
    Define the assignment costs when running detection -- truth assignment
    """
    if metric == "3D_IoU":
        try:
            cost = -det.box3d.IoU(tru.box3d)
        except AttributeError:
            try:
                cost = -det.IoU(tru)
            except AttributeError as e:
                raise AttributeError('Could not find a suitable attribute...did you want a 2D_IoU as metric?')
        if cost > -0.01:
            cost = np.inf
    elif metric == "center_dist":
        center1 = _get_center(det)
        center2 = _get_center(tru)
        if det.T_reference != tru.T_reference:  # convert to det frame
            T_switch = (det.T_reference @ tru.T_reference.T).T
            center2 = tru.T_reference.coordinates.convert(
                T_switch @ center2, det.T_reference.coordinates
            )
        if len(center1) == 2:
            # assume BEV center distance
            center2 = center2[[2, 0]]
            center2[1] *= -1
        cost = np.sum((center1 - center2) ** 2)
        if (radius2 is not None) and (cost > radius2):
            cost = np.inf
    elif metric == "2D_IoU":
        try:
            tru_box = tru.box2d
        except AttributeError as e:
            # Check if in view first
            if box_in_fov(tru.box3d, det.box2d.calibration):
                tru_box = tru.box3d.project_to_2d_bbox(
                    det.box2d.calibration, check_origin=True
                )
            else:
                return np.inf
        cost = -det.box2d.IoU(tru_box)
        if cost > -1e-8:
            cost = np.inf
    elif metric == "2D_FV_IoU":
        # project 3d box into 2D FV
        raise NotImplementedError
    elif metric == "2D_BEV_IoU":
        # project 3d box into 2D BEV
        raise NotImplementedError
    else:
        raise RuntimeError(
            "Cannot understand assignment metric...choices are: [3D_IoU, center_dist, 2D_IoU, 2D_FV_IoU, 2D_BEV_IoU]"
        )
    return cost


# def IOU_2d(corners1, corners2):
#     """Compute the IoU 2D

#     corners1, corners2 - an array of [xmin, ymin, xmax, ymax]
#     """
#     inter = bbox.box_intersection(corners1, corners2)
#     union = bbox.box_union(corners1, corners2)
#     return inter / union


# def IOU_3d(corners1, corners2):
#     """Compute the IoU 3D

#     corners1: numpy array (8,3), assume up direction is negative Y
#     """
#     inter = bbox.box_intersection(corners1, corners2)
#     union = bbox.box_union(corners1, corners2)
#     return inter / union
