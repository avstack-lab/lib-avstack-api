# @Author: Spencer Hallyburton <spencer>
# @Date:   2022-01-20
# @Filename: visualize.py
# @Last modified by:   spencer
# @Last modified time: 2022-01-20

import numpy as np
import quaternion
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.widgets import Slider, Button

try:
    from ipywidgets import interact, widgets
except ModuleNotFoundError as e:
    print('Cannot find ipywidgets...cannot run visualizations')

from avstack.geometry import bbox, NominalOriginCamera, NominalOriginStandard, StandardCoordinates
from avstack.objects import VehicleState
from avstack.modules.perception.detections import BoxDetection
from avapi.evaluation import color_from_object_type

from .base import show_lidar_bev_with_boxes, show_image_with_boxes


def _box_to_bev_rect(box_standard, color, facecolor=None, linewidth=3, linestyle='solid'):
    """NOTE: angle is with respect to the bottom left corner, NOT the center"""
    width = box_standard.l
    height = box_standard.w
    # -- create nominal box, rotate, translate
    bl_nom = np.array([-width/2, -height/2])
    yaw = box_standard.yaw
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw),  np.cos(yaw)]])
    left, bottom = R @ bl_nom + box_standard.t[:2]
    angle = box_standard.yaw * 180./np.pi  # angle w.r.t. bottom left corner
    fill = False if facecolor is None else True
    rect = plt.Rectangle((left, bottom), width, height, angle, alpha=0.9,
                         edgecolor=color, facecolor=facecolor, linewidth=linewidth,
                         linestyle=linestyle, fill=fill)
    return rect


def video_from_images(video_file, imgs, fps=10):
    print('making video...')
    height, width = imgs['fv'].shape[:2]
    fourcc=0
    video = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
    for idx in range(imgs['fv'].shape[3]):
        video.write(cv2.cvtColor(imgs['fv'][:,:,:,idx], cv2.COLOR_BGR2RGB))
    print('done making video')


def create_track_movie(extent, track_results_in,
        ego_box=None, object_linewidth=3, object_linestyle='solid',
        add_points=None, highlight_track_IDs=None, highlight_color='magenta',
        highlight_linewidth=2, highlight_linestyle='dashed', nframes=np.inf,
        inline=True, show_track_lines=True,
        show_track_pred=False, track_pred_future=2, track_pred_dt=0.1,
        projection=['bev'], fig_width=12, show_truth=False, save_video=False,
        video_file='track_movie.avi'):
    """Tracks without showing perception data
    extent in standard (lidar) coordinates
    """
    track_results = deepcopy(track_results_in)
    assert len(projection)==1, 'Can only do 1 projection for now'
    assert (add_points is None) or (len(add_points) == 0) or (len(add_points)==len(track_results)), len(add_points)
    if (add_points is not None) and (len(add_points) > 0):
        add_points_real = []
        for i, pts in enumerate(add_points):
            add_points_real.append([])
            for pt in pts:
                if (pt is not None) and (not isinstance(pt, str)):
                    pt.change_origin(NominalOriginStandard)
                    add_points_real[i].append(pt)
        add_points = add_points_real
    if (highlight_track_IDs is None) or (len(highlight_track_IDs) == 0):
        highlight_track_IDs = [[]] * len(track_results)
    else:
        assert (len(highlight_track_IDs)==len(track_results)), len(highlight_track_IDs)

    if ego_box is not None:
        ego_box.change_origin(NominalOriginStandard)

    # Get all track locations and predictions ahead of time
    trk_points = {}  # ID: frame: pt
    future_dts = np.arange(0, track_pred_future, track_pred_dt)
    trk_preds = {}  # ID: frame: dt_future: pt

    for idx, tr in track_results.items():
        for track in tr.tracks:
            if isinstance(track, VehicleState):
                track.change_origin(NominalOriginStandard)
                if track.ID not in trk_points:
                    trk_points[track.ID] = {}
                    trk_preds[track.ID] = {}
                if isinstance(track, (VehicleState, VehicleState)):
                    trk_points[track.ID][idx] = track.box3d.t
                    trk_preds[track.ID][idx] = {}
                    for dt_f in future_dts:
                        trk_preds[track.ID][idx][dt_f] = track.predict(dt_f).box3d.t
                else:
                    raise NotImplementedError(type(track))
        for truth in tr.truths:
            if isinstance(truth, VehicleState):
                truth.change_origin(NominalOriginStandard)

    def f(idx):
        axs_slider.clear()
        handles = []
        labels = []

        def plot_tracks(colors, tracks):
            for color, track in zip(colors, tracks):
                col = tuple(c/255.0 for c in color)
                do_highlight = isinstance(track, VehicleState) and (track.ID in highlight_track_IDs[idx])
                if isinstance(track, (VehicleState, BoxDetection)):
                    # -- full-fledged boxes (x=forward, y=left)
                    if 'bev' in projection:
                        rect = _box_to_bev_rect(track.box, col, linewidth=object_linewidth, linestyle=object_linestyle)
                        # add highlighting
                        if do_highlight:
                            fac = 1.2
                            box_enlarged = bbox.Box3D([fac*track.box.h, fac*track.box.w,
                                fac*track.box.l, track.position, track.box.attitude], track.position.origin)
                            axs_slider.add_patch(_box_to_bev_rect(box_enlarged, color=highlight_color, linestyle=highlight_linestyle, linewidth=highlight_linewidth))
                    elif 'fv' in projection:
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                    axs_slider.add_patch(rect)
                else:
                    if 'bev' in projection:
                        assert track.filter.x_vector.coordinates == StandardCoordinates
                        x_vec = track.filter.x_vector
                        axs_slider.scatter(x_vec[0], x_vec[1], s=10, color=col)
                    elif 'fv' in projection:
                        raise NotImplementedError
                    else:
                        raise NotImplementedError

        # -- show ego box
        if ego_box is not None:
            rect = _box_to_bev_rect(ego_box, 'black', facecolor='black')
            axs_slider.add_patch(rect)
            handles.append(rect)
            labels.append('Ego Bounding Box')

        # -- show true objects and tracks
        if idx in track_results:
            plot_tracks(track_results[idx].colors['detections'], track_results[idx].tracks)
            plot_tracks(track_results[idx].colors['truths'], track_results[idx].truths)

        # -- add lines
        if show_track_lines:
            hist = None
            for ID in trk_points:
                pts_bev = np.asarray([[pt[0], pt[1]] for idx_trk, pt in trk_points[ID].items() if idx_trk <= idx])
                if len(pts_bev) > 0:
                    hist, = axs_slider.plot(pts_bev[:,0], pts_bev[:,1], color='black', linewidth=3)
                    axs_slider.scatter(pts_bev[-1,0], pts_bev[-1,1], color='black', marker='x', s=30)
            if hist is not None:
                handles.append(hist)
                labels.append('Track History')

        # -- add predictions
        if show_track_pred:
            pred = None
            for ID in trk_preds:
                if idx in trk_preds[ID]:
                    pts_pred = np.asarray([[pt[0], pt[1]] for dt, pt in trk_preds[ID][idx].items()])
                    pred, = axs_slider.plot(pts_pred[:,0], pts_pred[:,1], color='black', linestyle='--', linewidth=3)
            if pred is not None:
                handles.append(pred)
                labels.append(f'{track_pred_future}s Track Prediction')

        # -- add objects
        if (add_points is not None) and (len(add_points) > 0):
            if add_points[idx] is not None:
                for pts in add_points[idx]:
                    if pts is not None:
                        axs_slider.scatter(pts[0], pts[1], marker='x', s=10, color='black')

        # -- show truths
        if show_truth:
            for color, truth in zip(track_results[idx].colors['truths'], track_results[idx].truths):
                col = tuple(c/255.0 for c in color)
                if 'bev' in projection:
                    truth.box3d.change_origin(NominalOriginStandard)
                    rect = _box_to_bev_rect(truth.box3d, col, linewidth=object_linewidth, linestyle=object_linestyle)
                elif 'fv' in projection:
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                axs_slider.add_patch(rect)

        # add fake elements to legend
        nominal_box = bbox.Box3D([2, 2, 4, np.zeros((3,)), np.quaternion(1)], NominalOriginStandard)
        # -- unsafe
        handles.append(_box_to_bev_rect(nominal_box, color=highlight_color, linewidth=highlight_linewidth, linestyle=highlight_linestyle))
        labels.append('Ego believes unsafe')
        # -- true positive
        col = tuple(c/255.0 for c in color_from_object_type('true_positive'))
        handles.append(_box_to_bev_rect(nominal_box, color=col, linewidth=object_linewidth, linestyle=object_linestyle))
        labels.append('Det: True Positive')
        # -- false positive
        col = tuple(c/255.0 for c in color_from_object_type('false_positive'))
        handles.append(_box_to_bev_rect(nominal_box, color=col, linewidth=object_linewidth, linestyle=object_linestyle))
        labels.append('Det: False Positive')
        # -- true positive
        col = tuple(c/255.0 for c in color_from_object_type('truth'))
        handles.append(_box_to_bev_rect(nominal_box, color=col, linewidth=object_linewidth, linestyle=object_linestyle))
        labels.append('Tru: True Positive')
        # -- false negative
        col = tuple(c/255.0 for c in color_from_object_type('false_negative'))
        handles.append(_box_to_bev_rect(nominal_box, color=col, linewidth=object_linewidth, linestyle=object_linestyle))
        labels.append('Tru: False Negative')

        # set limits
        axs_slider.set_xlim(extent[0])
        axs_slider.set_ylim(extent[1])
        axs_slider.set_xlabel('Forward (m)')
        axs_slider.set_ylabel('Left (m)')
        axs_slider.set_title("Ego-Centric Bird's Eye Track View - Frame %03i" % int(idx))
        axs_slider.legend(handles, labels, loc='lower right',  bbox_to_anchor=(1.21, 0))
        if inline:
            fig.canvas.draw()
        else:
            fig.canvas.draw_idle()
        if idx == 0:  # so we only adjust once
            plt.tight_layout()

    # make video
    fig_height = (extent[1][1] - extent[1][0]) / (extent[0][1] - extent[0][0]) * fig_width
    fig, axs_slider = plt.subplots(1, len(projection), figsize=(fig_width, fig_height))
    axs_slider.set_facecolor((220/255.0,220/255.0,220/255.0))
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width); height = int(height)
    if save_video:
        imgs = {'fv':[]}
        for i in range(int(min(nframes, len(track_results)))):
            f(i);
            img = np.fromstring(fig.canvas.tostring_rgb(),
            dtype='uint8').reshape(height, width, 3)
            imgs['fv'].append(img)
        imgs['fv'] = np.stack(imgs['fv'], axis=-1)
        video_from_images(video_file, imgs, fps=10)

    # make figure
    if inline:
        interact(f, idx=widgets.IntSlider(min=0, max=min(nframes-1, len(track_results)-1), step=1, value=0))
    else:
        ax_x = plt.axes([0.25, 0.1, 0.65, 0.03])
        slide = Slider( ax=ax_x,
                        label='Frame',
                        valmin=0,
                        valmax=len(track_results)-1,
                        valinit=0,
                        valstep=1.0,
                        orientation='horizontal')
        plt.subplots_adjust(bottom=0.25)
        slide.on_changed(f)
        f(0)
        plt.show()


def create_track_percep_movie(DM, track_results, iframe_start=0, nframes=np.inf, inline=True,
    projection=['fv'], sensor='image-2', show_truth=True, figsize=(14,8),
    save_video=False, video_file='track_percep_movie.avi'):
    """Uses 3D tracks and image data to create a video"""
    init = False
    count = 0
    idxs_record = np.asarray(sorted(list(track_results.keys())))
    if len(idxs_record) > 0:
        i_start = np.argmin(np.abs(idxs_record-iframe_start))
    else:
        i_start = iframe_start
    if isinstance(projection, str):
        projection = [projection]
    for proj in projection:
        assert proj in ['fv', 'bev'], 'Do not recognize %s projection' % proj
    imgs_ALL = {}

    # --- get images
    n_max = min(nframes, len(track_results)) if len(track_results) > 0 else nframes
    for i in range(n_max):
        # -- get real tracks
        if len(idxs_record) > 0:
            idx = idxs_record[i_start + i]
            calib = DM.get_calibration(idx, sensor)
            track_boxes = []
            for trk in track_results[idx].tracks:
                trk_copy = deepcopy(trk)
                trk_copy.change_origin(calib.origin)
                track_boxes.append(trk_copy)
            box_colors = track_results[idx].colors['detections']
            if show_truth:
                for truth in track_results[idx].truths:
                    truth.box.change_origin(calib.origin)
                    track_boxes.append(truth)
                box_colors.extend(track_results[idx].colors['truths'])
        else:
            idx = i_start + i
            calib = DM.get_calibration(idx, sensor)
            track_boxes = []
            box_colors = []

        # -- fv projection
        if 'fv' in projection:
            img = DM.get_image(idx, sensor=sensor)
            img3d = show_image_with_boxes(img, track_boxes,
                box_colors=box_colors, show=False, return_images=True)
            imgs_ALL['fv'] = img3d[...,None] if not init else np.concatenate((imgs_ALL['fv'], img3d[...,None]), axis=3)
        if 'bev' in projection:
            raise NotImplementedError
            # lidar = DM.get_lidar(idx)
            # imgbev = show_lidar_bev_with_boxes(lidar, calib=calib, labels=track_boxes, box_colors=box_colors, showbev=False, return_image=True)
            # imgs_ALL['bev'] = imgbev[...,None] if not init else np.concatenate((imgs_ALL['bev'], imgbev[...,None]), axis=3)
        init = True

    # --- make movie
    if inline:
        if save_video:
            video_from_images(video_file, imgs_ALL, fps=10)
        def f(idx):
            if len(projection) == 1:
                try:
                    axs_slider.imshow(imgs_ALL[projection[0]][:,:,:,idx])
                    fig.canvas.draw()
                except Exception as e:
                    print(imgs_ALL)
                    raise e
            else:
                for ax, proj in zip(axs_slider, projection):
                    ax.imshow(imgs_ALL[proj][:,:,:,idx])
                    fig.canvas.draw()
            plt.axis('off')

        fig, axs_slider = plt.subplots(1, len(projection), figsize=figsize)
        interact(f, idx=widgets.IntSlider(min=0, max=n_max-1, step=1, value=0))
    else:
        raise NotImplementedError
