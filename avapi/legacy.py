# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-08
# @Description:
"""

"""

import abc, os, sys, shutil, glob
import numpy as np
from tqdm import tqdm
from avstack.geometry import Box2D, Box3D, get_coordinates_from_string
import avstack.transformations as tforms


class Dataset():
    """
    Define a dataset of perception data

    The Dataset classes provide common methods for reading, writing, and
    manipulating perception data. These are not limited to objet detection
    datasets and can be extended to other datasets
    """
    name = ''
    subfolders_essential = []
    subfolders_optional = []
    subfolders_all = []

    def __init__(self, data_dir, split, labels_only=False):
        """to instantiate a particular dataset"""
        self.labels_only = labels_only
        self.data_dir = data_dir
        self.split = split
        self.split_path = os.path.join(self.data_dir, self.split)
        assert os.path.exists(self.split_path), '{} does not exist yet!!'.format(self.split_path)

    def __len__(self):
        return len(self.idxs)

    @classmethod
    def make_new(cls, data_dir, split):
        A = super().__new__(cls)
        A.__init__(data_dir, split)
        return A

    @property
    def idxs(self):
        """Wrapper to the classmethod using the split path"""
        return self._get_idxs_folder(self.split_path, self.labels_only)

    def check_idx(self, idx):
        assert idx in self.idxs, f'Candidate index, {idx}, not in index set {self.idxs}'

    @abc.abstractmethod
    def get_data(self, idx, folder=None):
        """Returns a dictionary with data for a frame"""

    @abc.abstractmethod
    def wipe_experiment(cls, path_to_folder, remove_tree=False):
        """Delete an experiment"""

    @abc.abstractmethod
    def copy_experiment_from_to(cls, src_base_path, dest_base_path, nframes=None, frame_list=None):
        """Copies an experiment from a source to dest folder"""

    @abc.abstractmethod
    def idx_to_prefix(idx):
        """Convert an idx (unique file ID) to a string"""

    @abc.abstractmethod
    def _get_idxs(path_to_folder):
        """Get the indices within a data folder"""

    @classmethod
    def _get_idxs_folder(cls, folder, labels_only=False):
        """Wrapper around get indices using sets"""
        idxs = None
        if labels_only:  # HACK!!!
            fdrs = ['label_2']
        else:
            fdrs = cls.subfolders_essential
        for subfolder in fdrs:
            full_path = os.path.join(folder, subfolder)
            if idxs is None:
                idxs = set(cls._get_idxs(full_path))
            else:
                idxs = idxs.intersection(set(cls._get_idxs(full_path)))
        return np.asarray(sorted(list(idxs)))

    @classmethod
    def _get_frame_list_from_nframes(cls, folder_path, nframes):
        all_idxs = cls._get_idxs_folder(folder_path)
        if nframes > len(all_idxs):
            print('Requested nframes ({}) is larger than number of frames ({}), so taking all'.format(nframes, len(all_idxs)))
            nframes = len(all_idxs)
        return all_idxs[:nframes]

    @classmethod
    def _wipe_data(cls, path_to_folder):
        """Delete specifically the data of an experiment"""
        cls._check_folder_before_bad_things(path_to_folder)
        if os.path.exists(path_to_folder):
            if os.path.islink(path_to_folder):
                os.unlink(path_to_folder)
            else:
                shutil.rmtree(path_to_folder)

    @classmethod
    def _copy_data_from_to(cls, src_base_path, dest_base_path, frame_list):
        """Copy specifically the data from an experiment"""
        # Get frame list
        all_idxs = cls._get_idxs_folder(src_base_path)
        if np.all([idx in frame_list for idx in all_idxs]):
            copy_all = True  # TODO: JUST MAKE THIS A SYMLINK INSTEAD (?)
        else:
            copy_all = False

        # Copy files included in the frame list to new destimation
        print('Copying each of %i subfolders' % len(cls.subfolders_all))
        for subfolder in tqdm(cls.subfolders_all):
            src_folder = os.path.join(src_base_path, subfolder)
            dest_folder = os.path.join(dest_base_path, subfolder)

            # Check if source exists
            if not os.path.exists(src_folder):
                continue

            # If copy all, use shutil, otherwise, copy one by one
            if os.path.exists(dest_folder):
                if os.path.islink(dest_folder):
                    os.unlink(dest_folder)
                else:
                    shutil.rmtree(dest_folder)
            if copy_all:
                shutil.copytree(src_folder, dest_folder)
            else:
                os.makedirs(dest_folder, exist_ok=True)
                for idx_f in frame_list:
                    prefix = cls.idx_to_prefix(idx_f)
                    fname = glob.glob(os.path.join(src_folder, '{}.*'.format(prefix)))

                    assert len(fname) == 1, 'Only one frame can match the glob'
                    shutil.copy(fname[0], fname[0].replace(src_folder, dest_folder))
        if os.path.exists(os.path.join(src_base_path, 'timestamps.txt')):
            shutil.copy(os.path.join(src_base_path, 'timestamps.txt'),
                        os.path.join(dest_base_path, 'timestamps.txt'))

    @staticmethod
    def _check_folder_before_bad_things(path_to_folder):
        """Ensure that we specially mark folders before we do things like delete"""
        acceptable_markers = ['experiment', 'TEST', 'temp']
        found_mark = False
        for mark in acceptable_markers:
            if mark in path_to_folder:
                found_mark = True
        if not found_mark:
            raise RuntimeError('Must have an acceptable marker word in folder name to proceed')


# ===============================================================
# OBJECT DATASETS
# ===============================================================

class Object3dDataset(Dataset):
    def get_data(self, idx):
        data = {}
        folder_names_inv = {v:k for k, v in self.folder_names.items()}
        for subfolder in self.subfolders_all:
            if os.path.exists(os.path.join(self.split_path, subfolder)):
                fname_inv = folder_names_inv[subfolder]
                data[fname_inv] = self.get_data_by_type(idx, fname_inv)
        return data

    def get_data_by_type(self, idx, data_string):
        if data_string.lower() in ['label']:
            return self.get_labels(idx)
        elif data_string.lower() in ['image']:
            return self.get_image(idx)
        elif data_string.lower() in ['lidar']:
            return self.get_lidar(idx)
        elif data_string.lower() in ['calibration']:
            return self.get_calibration(idx)
        else:
            raise NotImplementedError('Cannot return {} type'.format(data_string))

    def get_calibration(self, idx):
        raise NotImplementedError

    def get_image(self, idx):
        raise NotImplementedError

    @abc.abstractmethod
    def get_labels(self, idx):
        """This must be implemented"""

    def get_lidar(self, idx):
        raise NotImplementedError

    def view_data(self, idx, idx_label=None, lidar_extent=None, addbox=[], colormethod='depth'):
        image_good = ('image' in self.folder_names) and (os.path.exists(os.path.join(self.split_path, self.folder_names['image'])))
        lidar_good = ('lidar' in self.folder_names) and (os.path.exists(os.path.join(self.split_path, self.folder_names['lidar'])))
        if image_good:
            self.view_camera_data(idx, idx_label, addbox)
        else:
            print('Cannot show image with labels')
        if image_good and lidar_good:
            self.view_lidar_on_camera(idx, idx_label, colormethod)
        else:
            print('Cannot show lidar on image')
        if lidar_good:
            self.view_lidar_bev(idx, idx_label, lidar_extent, colormethod)
        else:
            print('Cannot show lidar bev')

    def view_camera_data(self, idx, idx_label=None, addbox=[]):
        from avstack import visualize
        calib = self.get_calibration(idx)
        img = self.get_image(idx)
        labels = self.get_labels(idx)
        if idx_label is not None:
            labels = labels[idx_label]
        visualize.show_image_with_boxes(img, labels, calib, inline=True, addbox=addbox)

    def view_lidar_on_camera(self, idx, idx_label, colormethod='depth'):
        from avstack import visualize
        img = self.get_image(idx)
        calib = self.get_calibration(idx)
        labels = self.get_labels(idx)
        if idx_label is not None:
            labels = labels[idx_label]
        lidar = self.get_lidar(idx)
        visualize.show_lidar_on_image(lidar, img, calib, inline=True, colormethod=colormethod)

    def view_lidar_bev(self, idx, idx_label, extent=None, colormethod='depth'):
        from avstack import visualize
        calib = self.get_calibration(idx)
        labels = self.get_labels(idx)
        if idx_label is not None:
            labels = labels[idx_label]
        lidar = self.get_lidar(idx)
        visualize.show_lidar_bev(lidar, extent=extent, calib=calib, labels=labels, colormethod=colormethod)