# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-08
# @Description:
"""

"""

import os, sys, glob, re, shutil
from cv2 import imread, imwrite
import numpy as np
import datetime

from avstack.geometry import GroundPlane, CameraCoordinates, Box3D
from ..utils import get_indices_in_folder, check_xor_for_none, get_timestamps
from ..dataset import Object3dDataset
from .calibration import KittiObjectCalibration
from .objects import KittiObject3dLabel
from . import  _parseTrackletXML as xmlParser


class KittiObjectDataset(Object3dDataset):
    name = 'KITTI'
    subfolders_essential = ['velodyne', 'image_2', 'calib', 'label_2']
    subfolders_optional = ['planes', 'velodyne_CROP', 'velodyne_reduced']
    subfolders_all = list(set(subfolders_essential).union(set(subfolders_optional)))
    folder_names  = {'image':'image_2', 'lidar':'velodyne', 'label':'label_2',
        'calibration':'calib', 'ground':'planes', 'disparity':'disparity'}

    CFG = {}
    CFG['num_lidar_features'] = 4
    CFG['IMAGE_WIDTH'] = 1242
    CFG['IMAGE_HEIGHT'] = 375
    CFG['IMAGE_CHANNEL'] = 3
    framerate = 10

    def __init__(self,  data_dir, split, labels_only=False):
        super().__init__(data_dir, split, labels_only)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'KittiObjectDataset at {self.split_path}'

    @classmethod
    def wipe_experiment(cls, path_to_folder, remove_tree=False):
        imset_path = cls._get_imset_path_from_data_path(path_to_folder)
        cls._wipe_imageset(imset_path)
        cls._wipe_data(path_to_folder)

    @classmethod
    def copy_experiment_from_to(cls, src_base_path, dest_base_path, nframes=None, frame_list=None):
        # Get frame list
        check_xor_for_none(nframes, frame_list)
        if frame_list is None:
            frame_list = cls._get_frame_list_from_nframes(src_base_path, nframes)

        # Run data copying
        cls._copy_data_from_to(src_base_path, dest_base_path, frame_list)
        imset_dest = cls._get_imset_path_from_data_path(dest_base_path)
        cls._write_imset(imset_dest, frame_list)

    def get_calibration(self, idx):
        self.check_idx(idx)
        calib_fname = os.path.join(self.split_path, self.folder_names['calibration'], '%06d.txt'%idx)
        return KittiObjectCalibration(calib_fname)

    @classmethod
    def save_calibration(cls, calib, path, idx):
        cls.write_calibration(calib.calib_dict, path, idx)

    @staticmethod
    def write_calibration(calib_dict, path, idx):
        calib_filename = os.path.join(path, '%06d.txt'%idx)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        with open(calib_filename, 'w') as txt_file:
            for key, values in calib_dict.items():
                txt_file.write('{}:'.format(key))
                for val in values:
                        txt_file.write(' {}'.format(val))
                txt_file.write('\n')

    def get_timestamps(self, data_type=None, idx=None):
        """Returns times for measurements"""
        if data_type is not None:
            assert data_type in ['velodyne']
        if idx is not None:
            assert data_type is not None
        t_path = os.path.join(self.split_path, 'timestamps.txt')
        assert os.path.exists(t_path), f'Timestamp file {t_path} does not exist'
        times = {}
        t_format = "%Y-%m-%d_%H:%M:%S.%f"
        with open(t_path, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(":", 1)
                times[key] = np.asarray([datetime.datetime.strptime(v[:-4], t_format).timestamp() for v in value.split()])
        if data_type is None:
            return times
        else:
            if idx is None:
                return times[data_type]
            else:
                return times[data_type][idx]

    def get_disparity(self, idx, is_depth=True):
        self.check_idx(idx)
        disp_fname = os.path.join(self.split_path, self.folder_names['disparity'], '%06d.{}')
        if is_depth:
            return np.squeeze(np.load(disp_fname.format('npy')))
        else:
            return imread(disp_fname.format('jpeg'))

    def get_ground(self, idx):
        """Returns the coefficients of the ground plane in the camera frame"""
        self.check_idx(idx)
        ground_fname = os.path.join(self.split_path, self.folder_names['ground'], '%06d.txt'%(idx))
        coeffs_string = [line.rstrip() for line in open(ground_fname)][3]
        ground_plane = np.asarray([float(st) for st in coeffs_string.split()])
        if ground_plane[1] > 0:
            ground_plane = -ground_plane
        ground_plane = ground_plane / np.linalg.norm(ground_plane[0:3])
        return GroundPlane(ground_plane, CameraCoordinates)

    @classmethod
    def save_ground(cls, ground, path, idx):
        ground_filename = os.path.join(path, '%06d.txt'%idx)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        ground_write = '# Plane\nWidth 4\nHeight 1\n'
        coeffstr = ''
        for i in range(len(ground)):
            coeffstr += ' %.6e'%ground[i]
        ground_write += coeffstr[1::]  # remove leading space
        with open(ground_filename, 'w') as txt_file:
            txt_file.write(ground_write)

    def get_image(self, idx):
        self.check_idx(idx)
        img_fname = os.path.join(self.split_path, self.folder_names['image'], '%06d.png'%idx)
        assert os.path.exists(img_fname), f'Cannot find requested image file at {img_fname}'
        return imread(img_fname)[:,:,::-1]

    @classmethod
    def save_image(cls, image, path, idx):
        image_filename = os.path.join(path, '%06d.png'%idx)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        imwrite(image_filename, image)

    def get_labels(self, idx, whitelist_types=['Car', 'Pedestrian', 'Cyclist'], ignore_types=['DontCare']):
        self.check_idx(idx)
        label_file = os.path.join(self.split_path, self.folder_names['label'], '%06d.txt'%idx)
        return self.get_labels_from_file(label_file, whitelist_types, ignore_types)

    @classmethod
    def save_labels(cls, labels, path, idx, add_label_folder):
        if add_label_folder:
            label_folder_path = os.path.join(path, cls.folder_names['label'])
        else:
            label_folder_path = path
        cls._save_labels(labels, label_folder_path, idx)

    @classmethod
    def _save_labels(cls, labels, path, idx):
        label_txt_list = [lab.write_text() for lab in labels]
        cls.write_label_text(label_txt_list, path, idx)

    @staticmethod
    def write_label_text(label_txt_list, path, idx):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, '%06d.txt'%idx)
        with open(fname, 'w') as f:
            first = True
            for ltxt in label_txt_list:
                if not first:
                    ltxt = '\n' + ltxt
                else:
                    first = False
                f.write(ltxt)

    @classmethod
    def get_labels_from_file(cls, label_fname, whitelist_types=['Car', 'Pedestrian', 'Cyclist'], ignore_types=['DontCare']):
        assert os.path.exists(label_fname), f'Cannot find requested label file at {label_fname}'
        with open(label_fname, 'r') as f:
            lines = [line.rstrip() for line in f]
        labels = [KittiObject3dLabel(line) for line in lines]
        if (whitelist_types == 'all') or ('all' in whitelist_types):
            labs = labels
        else:
            labs = [lab for lab in labels if (lab.obj_type not in ignore_types) & (lab.obj_type in whitelist_types)]
        labs = np.asarray(labs)
        return labs

    @classmethod
    def get_labels_from_directory(cls, dir, idxs=None):
        """Get all labels from a directory of text files"""
        # Loop over files
        glob_dir = glob.glob(os.path.join(dir, '*.txt'))
        glob_dir = sorted(glob_dir)
        idx_all = get_indices_in_folder(glob_dir, idxs)
        if len(idx_all) == 0:
            print('No labels to load')
            return [], []
        labels_all = []
        for idx in idx_all:
            label_fname = os.path.join(dir, '%06d.txt'%idx)
            labels = cls.get_labels_from_file(label_fname)
            labels_all.append(labels)
        idx_all, labels_all = (list(t) for t in zip(*sorted(zip(idx_all, labels_all))))
        return labels_all, idx_all

    @staticmethod
    def save_labels_to_folder(label_folder_path, labels, idx):
        if type(labels) != np.ndarray:
            labels = np.asarray(labels)
        if labels.size == 0:
            return
        label_txt_list = [lab.write_text() for lab in labels]
        if not os.path.isdir(label_folder_path):
            os.makedirs(label_folder_path, exist_ok=True)
        label_filename = os.path.join(label_folder_path, '%06d.txt'%(idx))
        with open(label_filename, 'w') as f:
            first = True
            for ltxt in label_txt_list:
                if not first:
                    ltxt = '\n' + ltxt
                f.write(ltxt)
                first = False

    @classmethod
    def get_track_instances_from_directory(cls, dir):
        """Get all tracks from a directory of text files"""
        glob_dir = glob.glob(os.path.join(dir, '*.txt'))
        glob_dir = sorted(glob_dir)
        idx_all = get_indices_in_folder(glob_dir, idxs)
        if len(idx_all) == 0:
            print('No tracks to load')
            return [], []
        tracks_all = []
        for idx in idx_all:
            track_fname = os.path.join(dir, '%06d.txt'%idx)
            track_instance = cls.get_track_instance_from_file(track_fname)
            tracks_all.append(track_instance)
        idx_all, tracks_all = (list(t) for t in zip(*sorted(zip(idx_all, tracks_all))))
        return tracks_all, idx_all

    @classmethod
    def get_track_instance_from_file(cls, track_fname, whitelist_types=['Car', 'Pedestrian', 'Cyclist'], ignore_types=['DontCare']):
        assert os.path.exists(track_fname), f'Cannot find requested track file at {track_fname}'
        with open(track_fname, 'r') as f:
            lines = [line.rstrip() for line in f]
        track_instance = []
        for line in lines:
            form = line.split()[0]
            if form == 'kitti':
                track_instance.append(KittiObjectTrackInstance(line.replace('kitti ', '')))
            elif form == 'avstack':
                track_instance.append(AvstackTrackInstance(line.replace('carla ', '')))
            else:
                raise NotImplementedError(line[0])
        if whitelist_types != 'all':
            track_instance = [trk for trk in track_instance if (trk.obj_type not in ignore_types) & (trk.obj_type in whitelist_types)]
        return np.asarray(track_instance)

    def get_lidar(self, idx, filter_front=True):
        self.check_idx(idx)
        lidar_fname = os.path.join(self.split_path, self.folder_names['lidar'], '%06d.bin'%idx)
        lidar = np.fromfile(lidar_fname, dtype=np.float32).reshape((-1, self.CFG['num_lidar_features']))
        if filter_front:
            return lidar[lidar[:,0]>0,:]
        else:
            return lidar

    def convert_lidar_to_ply(self, idx, folder):
        import open3d as o3d
        lidar = self.get_lidar(idx)
        points = lidar.reshape((-1, 4))[:, 0:3]
        o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        os.makedirs(folder, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(folder, "%06d.ply"%idx), o3d_pcd)

    @classmethod
    def save_lidar(cls, lidar, path, idx):
        lidar_filename = os.path.join(path, '%06d.bin'%idx)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        lidar.tofile(lidar_filename)

    @classmethod
    def _wipe_imageset(cls, imset_path):
        cls._check_folder_before_bad_things(imset_path)
        if os.path.exists(imset_path):
            print('removing: %s' % imset_path)
            os.remove(imset_path)

    @staticmethod
    def _write_imset(imset_path, frame_list):
        with open(imset_path, 'w') as f:
            for idx in frame_list:
                f.write('%06d\n' % int(idx))

    @staticmethod
    def idx_to_prefix(idx):
        return '%06d' % idx

    @staticmethod
    def _get_idxs(path_to_folder):
        """Gets indices of items in a folder by KITTI standard"""
        fnames = glob.glob(os.path.join(path_to_folder, '*.*'))
        return sorted([int(f.strip().split('/')[-1].split('.')[-2]) for f in fnames])

    @staticmethod
    def _get_imset_path_from_data_path(path_to_folder):
        exp_name = path_to_folder.split('/')[-1]
        imset_folder = os.path.join(*path_to_folder.split('/')[:-1], 'ImageSets')
        if path_to_folder[0] == '/':
            imset_folder = '/' + imset_folder
        assert os.path.exists(imset_folder), f'ImageSets folder must exist 1 level above the data (tried {imset_folder})'
        return os.path.join(imset_folder, exp_name +'.txt')


class KittiRawData():
    """Converting sequences of raw data to standard object data on the fly"""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @staticmethod
    def rename_file_for_kitti(folder, ext):
        for f in os.listdir(folder):
            if f.endswith(ext):
                os.rename(os.path.join(folder, f),
                          os.path.join(folder, '%06d'%int(f.split('.')[0]) + ext))

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
        assert os.path.exists(date_folder), 'Date folder does not exist!'
        pattern = re.compile(".*_[0-9]+_sync$")
        sequence_ids = []
        for item in os.listdir(date_folder):
            seq_folder = os.path.join(date_folder, item)
            if os.path.isdir(seq_folder) and pattern.match(seq_folder):
                if (not tracklets_req) or (os.path.exists(os.path.join(seq_folder, 'tracklet_labels.xml'))):
                    sequence_ids.append(item)
        return sorted(sequence_ids)

    def get_converted_exp_path(self, date, id_seq=None, idx_seq=None, tracklets_req=True, path_append=''):
        check_xor_for_none(id_seq, idx_seq)
        # ==== Get folder path
        if idx_seq is not None:
            ids_options = self.get_sequence_ids_at_date(date, tracklets_req=tracklets_req)
            id_seq = ids_options[idx_seq]
        seq_folder = os.path.join(self.data_dir, date, id_seq)
        assert os.path.exists(seq_folder), f'{seq_folder} does not exist'
        exp_path = os.path.join(self.data_dir, '../object', seq_folder.split('/')[-1]+path_append)
        return id_seq, exp_path, seq_folder

    def convert_sequence(self, date, id_seq=None, idx_seq=None, iframe_start=0, max_frames=None, max_time=None, tracklets_req=True, path_append=''):
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

        id_seq, exp_path, seq_folder = self.get_converted_exp_path(date, id_seq, idx_seq, tracklets_req, path_append)

        # ==== Convert format to standard KITTI format in new location
        KOD = get_dataset_class('KittiObjectDataset')
        if os.path.exists(exp_path):
            shutil.rmtree(exp_path)

        # -- timestamps -- do each in its own section
        timestamps = {'velodyne':[], 'image_0':[], 'image_1':[], 'image_2':[], 'image_3':[], 'label':[]}

        # -- image
        im_folder_src = os.path.join(seq_folder, 'image_02', 'data')
        im_folder_dest = os.path.join(exp_path, 'image_2')
        print('copying image data...')
        shutil.copytree(im_folder_src, im_folder_dest)
        self.rename_file_for_kitti(im_folder_dest, '.png')
        timestamps['image_0'] = get_timestamps(os.path.join(seq_folder, 'image_00'))
        timestamps['image_1'] = get_timestamps(os.path.join(seq_folder, 'image_01'))
        timestamps['image_2'] = get_timestamps(os.path.join(seq_folder, 'image_02'))
        timestamps['image_3'] = get_timestamps(os.path.join(seq_folder, 'image_03'))

        # -- lidar
        li_folder_src = os.path.join(seq_folder, 'velodyne_points', 'data')
        li_folder_dest = os.path.join(exp_path, 'velodyne')
        print('copying lidar data...')
        shutil.copytree(li_folder_src, li_folder_dest)
        self.rename_file_for_kitti(li_folder_dest, '.bin')
        timestamps['velodyne'] = get_timestamps(os.path.join(seq_folder, 'velodyne_points'))

        # -- calibration
        c2c = KittiObjectCalibration.read_calib_file(os.path.join(self.data_dir, date, 'calib_cam_to_cam.txt'))
        v2c = KittiObjectCalibration.read_calib_file(os.path.join(self.data_dir, date, 'calib_velo_to_cam.txt'))
        i2v = KittiObjectCalibration.read_calib_file(os.path.join(self.data_dir, date, 'calib_imu_to_velo.txt'))
        calibs = {}
        calibs['P0'] = c2c['P_rect_00']
        calibs['P1'] = c2c['P_rect_01']
        calibs['P2'] = c2c['P_rect_02']
        calibs['P3'] = c2c['P_rect_03']
        calibs['R0_rect'] = c2c['R_rect_00']
        calibs['Tr_velo_to_cam'] = \
            [v2c['R'][0], v2c['R'][1], v2c['R'][2], v2c['T'][0],
             v2c['R'][3], v2c['R'][4], v2c['R'][5], v2c['T'][1],
             v2c['R'][6], v2c['R'][7], v2c['R'][8], v2c['T'][2]]
        calibs['Tr_imu_to_velo'] = \
            [i2v['R'][0], i2v['R'][1], i2v['R'][2], i2v['T'][0],
             i2v['R'][3], i2v['R'][4], i2v['R'][5], i2v['T'][1],
             i2v['R'][6], i2v['R'][7], i2v['R'][8], i2v['T'][2]]
        print('copying calibration data...')
        nfiles = max([len(v) for _, v in timestamps.items()])
        for idx in range(nfiles):
            KOD.write_calibration(calibs, os.path.join(exp_path, 'calib'), idx)
        calib = KittiObjectCalibration(os.path.join(exp_path, 'calib', '%06d.txt'%idx))

        # -- labels
        print('copying label data...')
        labels = {ifile:[] for ifile in range(nfiles)}
        tracklets = xmlParser.parseXML(os.path.join(seq_folder, 'tracklet_labels.xml'))
        for itrk, trk in enumerate(tracklets):
            h, w, l = trk.size  # in camera's frame
            for trans, rot, state, occ, trunc, amtOcclusion, amtBorders, iframe in trk:
                # Create bboxes
                if trunc not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                    continue
                yaw = np.pi/2-rot[2]
                t_rect = calib.project_velo_to_rect(trans[:,None].T)
                box_3d = Box3D([h, w, l, t_rect, yaw], coordinates=CameraCoordinates)
                box_2d = box_3d.project_to_2d_bbox(calib=calib)
                # Add to dictionary
                alpha = (yaw - np.arctan2(trans[2], trans[1])) % (2.*np.pi)
                label_text = get_kitti_label_text_from_bbox(
                    trk.objectType, box_3d, box_2d, trunc, occ[0], alpha)
                labels[iframe].append(KittiObject3dLabel(label_text))
        frame_list = []
        for idx, labs in labels.items():
            frame_list.append(idx)
            KOD.save_labels(labs, exp_path, idx, True)

        # -- full tracks
        # TODO

        # -- save timestamps
        print('copying timestamp data...')
        with open(os.path.join(exp_path, 'timestamps.txt'), 'w') as f:
            for key, values in timestamps.items():
                f.write('{}:'.format(key))
                f.write(', '.join([v.replace(' ', '_') for v in values]))
                f.write('\n')


        # -- write imageset file
        print('writing imageset file...')
        imset_path = KittiObjectDataset._get_imset_path_from_data_path(exp_path)
        KittiObjectDataset._write_imset(imset_path, frame_list)

        print('done copying data! - sequence contains %i files' % nfiles)
        return exp_path



import numpy as np
from avstack import transformations as tforms
from ..calibration import CameraCalibration



class KittiCalibration():
    def __init__(self):
        pass



class KittiObjectCalibration(CameraCalibration):
    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)

        # Prep data for base initialization
        P = np.reshape(calibs['P2'], [3,4])
        V2C = np.reshape(calibs['Tr_velo_to_cam'], [3,4])
        C2V = tforms.inverse_rigid_trans(V2C)
        R0 = np.reshape(calibs['R0_rect'], [3,3])

        # Call base initialization
        T = None
        super().__init__(T, P)

        self.V2C = V2C
        self.C2V = C2V
        self.R0 = R0
        self.P = P


        # add the others for possible use later
        self.P0 = calibs['P0']
        self.P1 = calibs['P1']
        self.P2 = calibs['P2']
        self.P3 = calibs['P3']
        self.R0_rect = calibs['R0_rect']
        self.Tr_velo_to_cam = calibs['Tr_velo_to_cam']
        self.Tr_imu_to_velo = calibs['Tr_imu_to_velo']
        self.calib_dict = calibs

    @staticmethod
    def read_calib_file(filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    @staticmethod
    def read_calib_from_video(calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        raise NotImplementedError
        data = {}
        cam2cam = KittiObjectCalibration.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = KittiObjectCalibration.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3,4))
        Tr_velo_to_cam[0:3,0:3] = np.reshape(velo2cam['R'], [3,3])
        Tr_velo_to_cam[:,3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data


    def project_velo_to_ref(self, pts_3d_velo):
        """ Projects lidar coordinates to ref

        pts_3d_velo - nx3 or nx4 where 4th channel is intensity
        """
        pts_3d_velo_hom = tforms.cart2hom(pts_3d_velo[:,0:3]) # nx4
        pts_3d_ref = np.dot(pts_3d_velo_hom, np.transpose(self.V2C))
        if pts_3d_velo.shape[1] == 4:
            pts_3d_ref = np.hstack([pts_3d_ref, pts_3d_velo[:,3][:,None]])
        return pts_3d_ref


    def transform_points_to_ref(self, pts_3d_velo):
        """ Projects coordinates to ref

        pts_3d_velo - nx3 or nx4 where 4th channel is intensity
        """
        pts_3d_velo_hom = tforms.cart2hom(pts_3d_velo[:,0:3]) # nx4
        pts_3d_ref = np.dot(pts_3d_velo_hom, np.transpose(self.V2C))
        if pts_3d_velo.shape[1] == 4:
            pts_3d_ref = np.hstack([pts_3d_ref, pts_3d_velo[:,3][:,None]])
        return pts_3d_ref

    def project_ref_to_velo(self, pts_3d_ref):
        """ Projects ref to back to lidar

        pts_3d_ref - nx3 or nx4 where 4th channel is intensity
        """
        pts_3d_ref_hom = tforms.cart2hom(pts_3d_ref[:,0:3]) # nx4
        pts_3d_velo = np.dot(pts_3d_ref_hom, np.transpose(self.C2V))
        if pts_3d_ref.shape[1] == 4:
            pts_3d_velo = np.hstack([pts_3d_velo, pts_3d_ref[:,3][:,None]])
        return pts_3d_velo

    def project_rect_to_ref(self, pts_3d_rect):
        """ Projects camera rectangular to ref

        pts_3d_rect - nx3 or nx4 where 4th channel is intensity
        """
        pts_3d_ref = np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect[:,0:3])))
        if pts_3d_rect.shape[1] == 4:
            pts_3d_ref = np.hstack([pts_3d_ref, pts_3d_rect[:,3][:,None]])
        return pts_3d_ref

    def project_ref_to_rect(self, pts_3d_ref):
        """ Projects ref to camera rectangular

        pts_3d_rect - nx3 or nx4 where 4th channel is intensity
        """
        pts_3d_rect = np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref[:,0:3])))
        if pts_3d_ref.shape[1] == 4:
            pts_3d_rect = np.hstack([pts_3d_rect, pts_3d_ref[:,3][:,None]])
        return pts_3d_rect

    def project_rect_to_velo(self, pts_3d_rect):
        """ Projects rect to velo

        pts_3d_rect - nx3 or nx4 where 4th channel is intensity
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect[:,0:3])
        pts_3d_velo = self.project_ref_to_velo(pts_3d_ref)
        if pts_3d_rect.shape[1] == 4:
            pts_3d_velo = np.hstack([pts_3d_velo, pts_3d_rect[:,3][:,None]])
        return pts_3d_velo

    def project_velo_to_rect(self, pts_3d_velo):
        """ Projects velo to rect

        pts_3d_rect - nx3 or nx4 where 4th channel is intensity
        """
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo[:,0:3])
        pts_3d_rect = self.project_ref_to_rect(pts_3d_ref)
        if pts_3d_velo.shape[1] == 4:
            pts_3d_rect = np.hstack([pts_3d_rect, pts_3d_velo[:,3][:,None]])
        return pts_3d_rect

    def project_rect_to_image(self, pts_3d_rect):
        """ Projects rect to camera image

        pts_3d_rect - nx3 or nx4 where 4th channel is intensity
        """
        pts_3d_rect_hom = tforms.cart2hom(pts_3d_rect[:,0:3])
        pts_2d = np.dot(pts_3d_rect_hom, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        pts_2d_cam = pts_2d[:,0:2]
        if pts_3d_rect.shape[1] == 4:
            pts_2d_cam = np.hstack([pts_2d_cam, pts_3d_rect[:,3][:,None]])
        return pts_2d_cam

    def project_velo_to_image(self, pts_3d_velo):
        """ Projects velo to camera image

        pts_3d_rect - nx3 or nx4 where 4th channel is intensity
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo[:,0:3])
        pts_2d_img = self.project_rect_to_image(pts_3d_rect)
        if pts_3d_velo.shape[1] == 4:
            pts_2d_img = np.hstack([pts_2d_img, pts_3d_velo[:,3][:,None]])
        return pts_2d_img

    def project_rect_to_bev(self, pts_3d_rect):
        """ Projects rect to BEV representation

        In rect coordinates, y is "down" so is reduced
        """
        return np.delete(pts_3d_rect, 1, axis=1)

    def project_velo_to_bev(self, pts_3d_velo):
        """ Projects velo to BEV representation

        In velo coordinates, z is "up" so is reduced
        """
        return np.delete(pts_3d_velo, 2, axis=1)

    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)



import numpy as np
import avstack.transformations as tforms
from avstack.geometry import Box2D, Box3D, CameraCoordinates
from ..objects import Object3d2dLabel, ObjectTrackInstance


class KittiObject3dLabel(Object3d2dLabel):
    def __init__(self, label_file_line):
        """
        Kitti Object Label

        Data indices:
        0  - object type
        1  - truncation
        2  - occlusion
        3  - alpha (observation angle)
        4  - xmin for 2d box
        5  - ymin for 2d box
        6  - xmax for 2d box
        7  - ymax for 2d box
        8  - h for 3d box
        9  - w for 3d box
        10 - l for 3d box
        11 - x for 3d box loc in camera rect coords
        12 - y for 3d box loc in camera rect coords
        13 - z for 3d box loc in camera rect coords
        14 - yaw for 3d box (heading angle)
        15 - score (if applicable)
        """
        data = label_file_line.strip('\n').split(' ')
        if data[0] == 'kitti':
            data = data[1:]
        data[1:] = [float(x) for x in data[1:]]
        # KITTI data indices:
        # Get the box info
        obj_type = data[0]  # 'Car', 'Pedestrian', ...
        box2d = np.array([data[4], data[5], data[6], data[7]])
        t = np.array([data[11],data[12],data[13]])
        box3d = [data[8], data[9], data[10], t, data[14]]
        COORD = CameraCoordinates

        # Init the super
        box2d = Box2D(box2d)
        box3d = Box3D(box3d, COORD)
        super().__init__('KITTI', obj_type, box2d, box3d)

        # Kitti-Specific things
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]
        if len(data) == 15:
            self.score = data[14]
        else:
            self.score = None

    def write_text(self):
        '''Outputs a list of information to write itself to a file externally'''
        return get_kitti_label_text_from_bbox(self.obj_type, self.box3d, self.box2d,
                                              trunc=self.truncation,
                                              occ=self.occlusion,
                                              alpha=self.alpha)

    def get_difficulty(self):
        """
        Difficulty index:
        - 0 = easy
        - 1 = moderate
        - 2 = hard
        - 3 = not specified/impossible??
        """
        if (label.occlusion < 0) or (label.truncation < 0):
            raise RuntimeError('Truth fields must be filled for difficulty')

        if label.source.lower() == 'kitti':
            bbox_h = label.box2d.get_corners()[3] - label.box2d.get_corners()[1]
            occ = label.occlusion
            trunc = label.truncation

            # Easy cases
            if (bbox_h >= 40) and (occ == 0) and (trunc <= .15):
                diff = 'easy'
            elif (bbox_h >= 25) and (occ <= 1) and (trunc <= .30):
                diff = 'moderate'
            elif (bbox_h >= 25) and (occ <= 2) and (trunc <= .50):
                diff = 'hard'
            else:
                diff = 'undetermined'
        else:
            raise NotImplementedError
        return diff


def get_kitti_label_from_bbox(obj_type, box3d, box2d):
    return KittiObject3DLabel(get_kitti_label_text_from_bbox(obj_type, box3d, box2d))


def get_kitti_label_from_3d_bbox(obj_type, box3d, P):
    return get_kitti_label_from_bbox(obj_type, box3d, box3d.project_to_2d_bbox(P=P))


def get_kitti_label_text_from_bbox(obj_type, box3d, box2d, trunc=-1, occ=-1, alpha=-1):
    return get_kitti_label_text(obj_type=obj_type,
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
                                yaw=box3d.yaw)


def get_kitti_label_text(obj_type, truncation, occlusion, alpha, xmin, ymin, xmax, ymax, h, w, l, tx, ty, tz, yaw):
    return "kitti {obj_type:s} {truncation:.2f} {occlusion:.4f} {alpha:.2f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} {h:.2f} {w:.2f} {l:.2f} {tx:.2f} {ty:.2f} {tz:.2f} {yaw:.2f}".format(obj_type=obj_type, truncation=truncation, occlusion=occlusion, alpha=alpha, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, h=h, w=w, l=l, tx=tx, ty=ty, tz=tz, yaw=yaw)



class KittiObjectTrackInstance(ObjectTrackInstance):
    def __init__(self, track_file_line):
        """
        Kitti Object Track

        Data indices:
        0 - frame
        1 - timestamp
        2 - track ID
        3 - object type
        4 - alpha
        5 - xmin
        6 - ymin
        7 - xmax
        8 - ymax
        9 - h
        10 - w
        11 - l
        12 - tx
        13 - ty
        14 - tz
        15 - yaw
        16 - score
        """
        data = track_file_line.strip('\n').split(' ')
        self.frame, timestamp, ID, obj_type = int(data[0]), float(data[1]), int(data[2]), data[3]
        super().__init__('kitti', obj_type, ID, timestamp)
        self.box3d = Box3D([float(d) for d in data[9:16]], CameraCoordinates)
        self.box2d = Box2D([int(float(d)) for d in data[5:9]])
        lab_txt = get_kitti_label_text_from_bbox(obj_type, self.box3d, self.box2d, alpha=float(data[4]))
        self.label = KittiObject3dLabel(lab_txt)
        self.score = float(data[16])

    @property
    def box(self):
        return self.box3d

    @property
    def coordinates(self):
        return self.box3d.coordinates

    @property
    def position(self):
        return self.box3d.t