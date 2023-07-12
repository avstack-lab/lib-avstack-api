# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-08-08
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-29
# @Description:
"""

"""
import glob
import json
import os
import re
import shutil
from datetime import datetime

import numpy as np
from avstack import calibration
from avstack.environment.objects import VehicleState
from avstack.geometry import (
    Attitude,
    GlobalOrigin3D,
    Position,
    ReferenceFrame,
    Rotation,
    Vector,
    bbox,
    q_mult_vec,
    q_stan_to_cam,
)
from avstack.geometry import transformations as tforms
from avstack.utils import check_xor_for_none
from cv2 import imread
from tqdm import tqdm

from .._dataset import BaseSceneDataset, BaseSceneManager
from . import _parseTrackletXML as xmlParser


def get_timestamps(src_folder):
    ts = []
    with open(os.path.join(src_folder, "timestamps.txt")) as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            ts.append(datetime.strptime(line[:-3], "%Y-%m-%d %H:%M:%S.%f"))
    return ts


def get_kitti_label_text_from_bbox(obj_type, box3d, box2d, trunc=-1, occ=-1, alpha=-1):
    return get_kitti_label_text(
        obj_type=obj_type,
        truncation=trunc,
        occlusion=occ,
        alpha=alpha,
        xmin=box2d.xmin,
        ymin=box2d.ymin,
        xmax=box2d.xmax,
        ymax=box2d.ymax,
        h=box3d.h,
        w=box3d.w,
        l=box3d.l,
        tx=box3d.t[0],
        ty=box3d.t[1],
        tz=box3d.t[2],
        yaw=box3d.yaw,
    )


def get_kitti_label_text(
    obj_type,
    truncation,
    occlusion,
    alpha,
    xmin,
    ymin,
    xmax,
    ymax,
    h,
    w,
    l,
    tx,
    ty,
    tz,
    yaw,
):
    return "kitti {obj_type:s} {truncation:.2f} {occlusion:.4f} {alpha:.2f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} {h:.2f} {w:.2f} {l:.2f} {tx:.2f} {ty:.2f} {tz:.2f} {yaw:.2f}".format(
        obj_type=obj_type,
        truncation=truncation,
        occlusion=occlusion,
        alpha=alpha,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        h=h,
        w=w,
        l=l,
        tx=tx,
        ty=ty,
        tz=tz,
        yaw=yaw,
    )


_nominal_whitelist_types = ["Car", "Pedestrian", "Cyclist"]
_nominal_ignore_types = ["DontCare", "Van", "Truck"]


class KittiObjectDataset(BaseSceneDataset):
    NAME = "KITTI"
    subfolders_essential = ["velodyne", "image_2", "calib", "label_2"]
    subfolders_optional = ["planes", "velodyne_CROP", "velodyne_reduced"]
    subfolders_all = list(set(subfolders_essential).union(set(subfolders_optional)))
    CFG = {}
    CFG["num_lidar_features"] = 4
    CFG["IMAGE_WIDTH"] = 1242
    CFG["IMAGE_HEIGHT"] = 375
    CFG["IMAGE_CHANNEL"] = 3
    img_shape = (CFG["IMAGE_HEIGHT"], CFG["IMAGE_WIDTH"], CFG["IMAGE_CHANNEL"])
    framerate = 10
    folder_names = {
        "image-2": "image_2",
        "lidar": "velodyne",
        "label": "label_2",
        "calibration": "calib",
        "ground": "planes",
        "disparity": "disparity",
        "timestamps": "timestamps",
    }
    data_endings = {
        "image-2": "png",
        "lidar": "bin",
        "label": "txt",
        "calibration": "txt",
    }
    sensors = {"main_lidar": "lidar", "lidar": "lidar", "main_camera": "image-2"}
    cameras = {"image-2"}
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types
    ego_size = [1.49, 1.82, 4.77]  # h, w, l
    sensor_IDs = {"lidar": 0, "image-0": 0, "image-1": 1, "image-2": 2, "image-3": 3}

    def __init__(
        self,
        data_dir,
        split,
        whitelist_types=_nominal_whitelist_types,
        ignore_types=_nominal_ignore_types,
    ):
        self.data_dir = data_dir
        self.split_path = os.path.join(data_dir, split)
        self.sequence_id = split
        super().__init__(whitelist_types, ignore_types)

    @property
    def frames(self):
        """Wrapper to the classmethod using the split path"""
        return self._get_frames_folder(self.split_path)

    def _load_frames(self, sensor: str):
        return self.frames

    def _load_ego(self, frame):
        """Ego origin is defined as the center of the vehicle"""
        reference = self._load_ego_reference(frame)
        ego = VehicleState("car")
        h, w, l = self.ego_size
        pos = Position(np.zeros((3,)), reference)
        rot = Attitude(np.quaternion(1), reference)
        box = bbox.Box3D(pos, rot, [h, w, l], where_is_t="bottom")
        ego.set(self.get_timestamp(frame), pos, box, attitude=rot)
        return ego

    @classmethod
    def _get_frames_folder(cls, folder):
        """Wrapper around get indices using sets"""
        frames = None
        fdrs = cls.subfolders_essential
        for subfolder in fdrs:
            full_path = os.path.join(folder, subfolder)
            if os.path.exists(full_path):
                if frames is None:
                    frames = set(cls._get_frames(full_path))
                else:
                    frames = frames.intersection(set(cls._get_frames(full_path)))
        if frames is None:
            print("No indices available for %s!" % folder)
            return None
        else:
            return np.asarray(sorted(list(frames)))

    @staticmethod
    def _get_frames(path_to_folder):
        """Gets indices of items in a folder by KITTI standard"""
        fnames = glob.glob(os.path.join(path_to_folder, "*.*"))
        return sorted([int(f.strip().split("/")[-1].split(".")[-2]) for f in fnames])

    @staticmethod
    def write_calibration(calib_dict, path, idx):
        calib_filename = os.path.join(path, "%06d.txt" % idx)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        with open(calib_filename, "w") as txt_file:
            for key, values in calib_dict.items():
                txt_file.write("{}:".format(key))
                for val in values:
                    txt_file.write(" {}".format(val))
                txt_file.write("\n")

    def _load_ego_reference(self, frame):
        return GlobalOrigin3D  # TODO: this is not ideal...

    def _load_calibration(self, frame, sensor, ego_reference):
        calib_fname = os.path.join(
            self.split_path, self.folder_names["calibration"], "%06d.txt" % frame
        )
        return self._load_calibration_from_file(
            calib_fname, sensor, self.img_shape, ego_reference
        )

    @staticmethod
    def _load_calibration_from_file(
        file, sensor, img_shape, ego_reference, ego_height=1.49
    ):
        if sensor == "labels":
            sensor = "image-2"
        calib_data = KittiObjectDataset.read_dict_text_file(file)
        if "R0_rect" not in calib_data:
            R_ref_2_C0 = np.eye(3)
        else:
            R_ref_2_C0 = np.reshape(calib_data["R0_rect"], [3, 3])  # transpose or no?
        q_ref_2_C0 = tforms.transform_orientation(R_ref_2_C0, "dcm", "quat")
        if "image" in sensor.lower():
            px = 0  # approximately centered forward-back
            pz = 1.65 - ego_height / 2  # relative to car
            if sensor.lower() == "image-0":
                py = 0  # approximately centered left-right
            elif sensor.lower() == "image-1":
                py = -0.54
            elif sensor.lower() == "image-2":
                py = 0.06
            elif sensor.lower() == "image-3":
                py = -0.48
            else:
                raise NotImplementedError(sensor.lower())
            x_O_2_C_in_O = np.array([px, py, pz])
            q_O_2_ref = q_stan_to_cam
            q_O_2_C = q_ref_2_C0 * q_O_2_ref
            reference = ReferenceFrame(x_O_2_C_in_O, q_O_2_C, ego_reference)
            P = np.reshape(
                calib_data["P%s" % sensor.lower().replace("image-", "")], [3, 4]
            )
            calib = calibration.CameraCalibration(
                reference, P, img_shape, channel_order="bgr"
            )
        elif "lidar" in sensor.lower():
            # -- get transform from cam 0 to lidar
            R_velo_to_cam0 = np.reshape(calib_data["Tr_velo_to_cam"], [3, 4])[:3, :3]
            q_C0_2_L = tforms.transform_orientation(
                R_velo_to_cam0, "dcm", "quat"
            ).conjugate()
            # -- get transform from origin to camera
            q_O_2_ref = q_stan_to_cam
            q_O_2_C0 = q_ref_2_C0 * q_O_2_ref
            # -- combine
            x_O_2_L_in_O = np.array([-0.27, 0, 1.73 - ego_height / 2])
            q_O_2_L = q_C0_2_L * q_O_2_C0
            calib = calibration.Calibration(
                ReferenceFrame(x_O_2_L_in_O, q_O_2_L, ego_reference)
            )
        else:
            raise NotImplementedError(sensor)
        return calib

    def _load_image(self, frame, sensor):
        if isinstance(sensor, str) and "image" in sensor:
            img_fname = os.path.join(
                self.split_path, self.folder_names[sensor], "%06d.png" % frame
            )
        else:
            img_fname = os.path.join(
                self.split_path,
                self.folder_names["image-%i" % sensor],
                "%06d.png" % frame,
            )
        return imread(img_fname)

    def _load_lidar(self, frame, sensor, **kwargs):
        filter_front = True  # always filter KITTI to front-only
        lidar_fname = os.path.join(
            self.split_path, self.folder_names["lidar"], "%06d.bin" % frame
        )
        lidar = np.fromfile(lidar_fname, dtype=np.float32).reshape(
            (-1, self.CFG["num_lidar_features"])
        )
        if filter_front:
            return lidar[lidar[:, 0] > 0, :]
        else:
            return lidar

    def _load_objects(
        self,
        frame,
        sensor="image-2",
        whitelist_types=["Car", "Pedestrian", "Cyclist"],
        ignore_types=["DontCare"],
    ):
        label_fname = os.path.join(
            self.split_path, self.folder_names["label"], "%06d.txt" % frame
        )
        object_calib = self.get_calibration(frame, sensor)
        objects = self._load_objects_from_file(
            label_fname, whitelist_types, ignore_types
        )
        for obj in objects:
            obj.change_reference(object_calib.reference, inplace=True)
        return objects

    def _load_timestamp(self, frame, utime=False, sensor="lidar"):
        if os.path.exists(
            os.path.join(self.split_path, self.folder_names["timestamps"])
        ):
            ts_fname = os.path.join(
                self.split_path, self.folder_names["timestamps"], "%06d.txt" % frame
            )
            ts = self.read_dict_text_file(ts_fname)[sensor]
        else:
            ts = frame * 0.1
        return ts

    def _save_calibration(self, frame, calib, folder, sensor_ID=2):
        calib_text_str = "P{}: {}".format(
            sensor_ID, " ".join([str(p) for p in np.reshape(calib.P, -1)])
        )
        fname = os.path.join(folder, "%06d.txt" % frame)
        with open(fname, "w") as f:
            f.write(calib_text_str)

    def _save_objects(self, frame, objects, folder, file):
        if file is None:
            file = os.path.join(folder, "%06d.txt" % frame)
        with open(file, "w") as f:
            str_list = [obj.encode() for obj in objects]
            for line in str_list:
                f.write(f"{line}\n")

    def _get_sensor_file_name(self, frame, sensor: str):
        return os.path.join(
            self.split_path,
            self.folder_names[sensor],
            "%06d.%s" % (frame, self.data_endings[sensor]),
        )

    @staticmethod
    def _get_imset_path_from_data_path(path_to_folder):
        exp_name = path_to_folder.split("/")[-1]
        imset_folder = os.path.join(*path_to_folder.split("/")[:-1], "ImageSets")
        if path_to_folder[0] == "/":
            imset_folder = "/" + imset_folder
        assert os.path.exists(
            imset_folder
        ), f"ImageSets folder must exist 1 level above the data (tried {imset_folder})"
        return os.path.join(imset_folder, exp_name + ".txt")

    @staticmethod
    def _write_imset(imset_path, frame_list):
        with open(imset_path, "w") as f:
            for idx in frame_list:
                f.write("%06d\n" % int(idx))


# ===============================================================
# LONGITUDINAL
# ===============================================================


class KittiScenesManager(BaseSceneManager):
    """Managing the raw dataset"""

    NAME = "Kitti"

    def __init__(self, data_dir, raw_data_dir=None, convert_raw=False, verbose=False):
        self.verbose = verbose
        self.data_dir = data_dir

        # scenes are tuples of (date, index) mapped to numbers
        self.raw_data_dir = raw_data_dir
        self.date_names = ["2011_09_26"]
        self.scene_tuples = [(i, j) for i in range(1) for j in range(46)]
        if convert_raw:
            self.convert_raw_dataset()

        # Process scenes
        self.scene_name_to_index = {sc: i for i, sc in enumerate(self.scene_tuples)}
        self.index_to_scene = {i: sc for i, sc in enumerate(self.scene_tuples)}

        # Get scenes
        self.scenes = [
            glob.glob(os.path.join(self.data_dir, date + "*"))
            for date in self.date_names
        ]
        self.scenes = sorted(
            [
                scene.replace(data_dir, "").lstrip("/")
                for scenes_this in self.scenes
                for scene in scenes_this
            ]
        )
        self.splits_scenes = self.make_splits_scenes(modval=4, seed=1)

    def convert_raw_dataset(self):
        if self.raw_data_dir is not None:
            KRD = KittiRawDataset(self.raw_data_dir)
            print("Converting dates and sequences from raw data")
            for idx_date, idx_seq in tqdm(self.scene_tuples):
                _ = KRD.convert_sequence(
                    self.date_names[idx_date],
                    idx_seq=idx_seq,
                    max_frames=None,
                    max_time=None,
                    tracklets_req=True,
                )
        else:
            print("No raw data dir available to postprocess...assuming already done")

    def get_scene_dataset_by_name(self, scene_name):
        return KittiObjectDataset(self.data_dir, scene_name)

    def get_scene_dataset_by_index(self, scene_idx):
        return self.get_scene_dataset_by_name(self.scenes[scene_idx])


class KittiRawDataset:
    """Converting sequences of raw data to standard object data on the fly"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    @staticmethod
    def rename_file_for_kitti(folder, ext):
        for f in os.listdir(folder):
            if f.endswith(ext):
                os.rename(
                    os.path.join(folder, f),
                    os.path.join(folder, "%06d" % int(f.split(".")[0]) + ext),
                )

    def get_available_dates(self):
        """Get all capture dates from data folder"""
        pattern = re.compile(".*[0-9]+_[0-9]+_[0-9]+$")
        dates = []
        for item in os.listdir(self.data_dir):
            dirname = os.path.join(self.data_dir, item)
            if os.path.isdir(dirname) and pattern.match(dirname):
                dates.append(item)
        return sorted(dates)

    def get_sequence_ids_at_date(self, date, tracklets_req=False):
        """
        :date - the date (as string) of the sequence capture
        :tracklets_req - boolean on if it is required that the tracklet xml exists
        """
        date_folder = os.path.join(self.data_dir, date)
        assert os.path.exists(date_folder), "Date folder does not exist!"
        pattern = re.compile(".*_[0-9]+_sync$")
        sequence_ids = []
        for item in os.listdir(date_folder):
            seq_folder = os.path.join(date_folder, item)
            if os.path.isdir(seq_folder) and pattern.match(seq_folder):
                if (not tracklets_req) or (
                    os.path.exists(os.path.join(seq_folder, "tracklet_labels.xml"))
                ):
                    sequence_ids.append(item)
        return sorted(sequence_ids)

    def get_converted_exp_path(
        self, date, id_seq=None, idx_seq=None, tracklets_req=True, path_append=""
    ):
        check_xor_for_none(id_seq, idx_seq)
        # ==== Get folder path
        if idx_seq is not None:
            ids_options = self.get_sequence_ids_at_date(
                date, tracklets_req=tracklets_req
            )
            try:
                id_seq = ids_options[idx_seq]
            except IndexError as e:
                raise IndexError(
                    "list index out of range...have you downloaded the tracklets to go with scenes?"
                )
        seq_folder = os.path.join(self.data_dir, date, id_seq)
        assert os.path.exists(seq_folder), f"{seq_folder} does not exist"
        exp_path = os.path.join(
            self.data_dir, "../object", seq_folder.split("/")[-1] + path_append
        )
        return id_seq, exp_path, seq_folder

    def convert_sequence(
        self,
        date,
        id_seq=None,
        idx_seq=None,
        iframe_start=0,
        max_frames=None,
        max_time=None,
        tracklets_req=True,
        path_append="",
        verbose=True,
    ):
        """
        :date - the date (as string) of the sequence capture
        :id_seq - the id (as string) of the sequence
        :idx_seq - the index (as integer) of the sequence to use
        """

        if iframe_start != 0:
            raise NotImplementedError
        if max_frames is not None:
            raise NotImplementedError
        if max_time is not None:
            raise NotImplementedError

        id_seq, exp_path, seq_folder = self.get_converted_exp_path(
            date, id_seq, idx_seq, tracklets_req, path_append
        )

        # ==== Convert format to standard KITTI format in new location
        KOD = KittiObjectDataset
        if os.path.exists(exp_path):
            shutil.rmtree(exp_path)

        # -- timestamps -- do each in its own section
        timestamps = {
            "velodyne": [],
            "image_0": [],
            "image_1": [],
            "image_2": [],
            "image_3": [],
            "label": [],
        }

        # -- image
        im_folder_src = os.path.join(seq_folder, "image_02", "data")
        im_folder_dest = os.path.join(exp_path, "image_2")
        if verbose:
            print("copying image data...")
        shutil.copytree(im_folder_src, im_folder_dest)
        self.rename_file_for_kitti(im_folder_dest, ".png")
        timestamps["image_0"] = get_timestamps(os.path.join(seq_folder, "image_00"))
        timestamps["image_1"] = get_timestamps(os.path.join(seq_folder, "image_01"))
        timestamps["image_2"] = get_timestamps(os.path.join(seq_folder, "image_02"))
        timestamps["image_3"] = get_timestamps(os.path.join(seq_folder, "image_03"))

        # -- lidar
        li_folder_src = os.path.join(seq_folder, "velodyne_points", "data")
        li_folder_dest = os.path.join(exp_path, "velodyne")
        if verbose:
            print("copying lidar data...")
        shutil.copytree(li_folder_src, li_folder_dest)
        self.rename_file_for_kitti(li_folder_dest, ".bin")
        timestamps["velodyne"] = get_timestamps(
            os.path.join(seq_folder, "velodyne_points")
        )

        # -- calibration
        c2c = KittiObjectDataset.read_dict_text_file(
            os.path.join(self.data_dir, date, "calib_cam_to_cam.txt")
        )
        v2c = KittiObjectDataset.read_dict_text_file(
            os.path.join(self.data_dir, date, "calib_velo_to_cam.txt")
        )
        i2v = KittiObjectDataset.read_dict_text_file(
            os.path.join(self.data_dir, date, "calib_imu_to_velo.txt")
        )
        calibs = {}
        calibs["P0"] = c2c["P_rect_00"]
        calibs["P1"] = c2c["P_rect_01"]
        calibs["P2"] = c2c["P_rect_02"]
        calibs["P3"] = c2c["P_rect_03"]
        calibs["R0_rect"] = c2c["R_rect_00"]
        calibs["Tr_velo_to_cam"] = [
            v2c["R"][0],
            v2c["R"][1],
            v2c["R"][2],
            v2c["T"][0],
            v2c["R"][3],
            v2c["R"][4],
            v2c["R"][5],
            v2c["T"][1],
            v2c["R"][6],
            v2c["R"][7],
            v2c["R"][8],
            v2c["T"][2],
        ]
        calibs["Tr_imu_to_velo"] = [
            i2v["R"][0],
            i2v["R"][1],
            i2v["R"][2],
            i2v["T"][0],
            i2v["R"][3],
            i2v["R"][4],
            i2v["R"][5],
            i2v["T"][1],
            i2v["R"][6],
            i2v["R"][7],
            i2v["R"][8],
            i2v["T"][2],
        ]
        if verbose:
            print("copying calibration data...")
        nfiles = max([len(v) for _, v in timestamps.items()])
        for idx in range(nfiles):
            KOD.write_calibration(calibs, os.path.join(exp_path, "calib"), idx)
        KDM = KOD("/".join(exp_path.split("/")[:-1]), exp_path.split("/")[-1])
        calib_image = KDM.get_calibration(idx, "image-2")
        calib_lidar = KDM.get_calibration(idx, "lidar")

        # -- labels
        if verbose:
            print("copying label data...")
        objects = {ifile: [] for ifile in range(nfiles)}
        tracklets = xmlParser.parseXML(
            os.path.join(seq_folder, "tracklet_labels.xml"), verbose=verbose
        )
        for itrk, trk in enumerate(tracklets):
            h, w, l = trk.size  # in camera's frame
            for trans, rot, state, occ, trunc, amtOcclusion, amtBorders, iframe in trk:
                # Create bboxes
                if trunc not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                    continue
                # use yaw=0 is forward which is not the KITTI standard
                yaw = rot[2]
                x_L_2_obj_in_L = trans
                q_L_2_obj = tforms.transform_orientation([0, 0, yaw], "euler", "quat")

                # convert to camera 2 frame
                obj_reference = calib_image.reference
                # -- quaternion
                q_O_2_L = calib_lidar.reference.q
                q_C2_2_O = calib_image.reference.q.conjugate()
                q_C2_2_L = q_O_2_L * q_C2_2_O
                q_C2_2_obj = q_L_2_obj * q_C2_2_L

                # -- translation
                q_L_2_C2 = q_C2_2_L.conjugate()
                x_L_2_obj_in_C2 = q_mult_vec(q_L_2_C2, x_L_2_obj_in_L)
                x_C2_2_O_in_O = -calib_image.reference.x
                x_O_2_L_in_O = calib_lidar.reference.x
                x_C2_2_L_in_O = x_O_2_L_in_O + x_C2_2_O_in_O
                x_C2_2_L_in_C2 = q_mult_vec(q_C2_2_O.conjugate(), x_C2_2_L_in_O)
                x_C2_2_obj_in_C2 = x_L_2_obj_in_C2 + x_C2_2_L_in_C2

                # Store fields
                pos = Position(x_C2_2_obj_in_C2, obj_reference)
                rot = Attitude(q_C2_2_obj, obj_reference)
                hwl = [h, w, l]
                box3d = bbox.Box3D(pos, rot, hwl, where_is_t="bottom")
                obj = VehicleState(trk.objectType, itrk)
                ts = timestamps["velodyne"][iframe].timestamp()
                vel = None
                acc = None
                ang = None
                obj.set(ts, pos, box3d, vel, acc, rot, ang)
                objects[iframe].append(obj)
        frame_list = []
        for idx, objs in objects.items():
            frame_list.append(idx)
            KDM.save_objects(idx, objs, os.path.join(exp_path, "label_2"))

        # -- full tracks
        # TODO

        # -- save timestamps
        if verbose:
            print("copying timestamp data...")
        with open(os.path.join(exp_path, "timestamps.txt"), "w") as f:
            for key, values in timestamps.items():
                f.write("{}:".format(key))
                f.write(", ".join([str(v.timestamp()) for v in values]))
                # f.write(', '.join([v.replace(' ', '_') for v in values]))
                f.write("\n")

        # -- write imageset file
        if verbose:
            print("writing imageset file...")
        imset_path = KittiObjectDataset._get_imset_path_from_data_path(exp_path)
        KittiObjectDataset._write_imset(imset_path, frame_list)

        if verbose:
            print("done copying data! - sequence contains %i files" % nfiles)
        obj_path = exp_path.replace("ImageSets/", "").replace(".txt", "")
        return os.path.realpath(obj_path)
