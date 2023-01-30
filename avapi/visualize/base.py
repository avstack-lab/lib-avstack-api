# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-19
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-09-06
# @Description:
"""

"""

import numpy as np
import cv2
from PIL import Image
from copy import copy, deepcopy

from avapi.evaluation import parse_color_string
from avstack.geometry import bbox, Box2D, Box3D, NominalOriginStandard
from avstack.modules.perception.detections import BoxDetection, MaskDetection
from avstack.datastructs import DataContainer
from avstack.objects import VehicleState
from avstack.utils import maskfilters

import matplotlib.pyplot as plt

lidar_cmap = plt.cm.get_cmap('hsv', 256)
lidar_cmap = np.array([lidar_cmap(i) for i in range(256)])[:,:3]*255


def get_lidar_color(value, mode='depth'):
    # Min range = 0 --> 1
    # Max range = 100 --> 255
    if mode == 'depth':
        scaling = 100
    elif mode == 'confidence':
        scaling = 2
    elif mode == 'randint':
        scaling = 50
    else:
        raise NotImplementedError(mode)

    idx = max(1, min(255, 255 * value/scaling))
    color = lidar_cmap[int(idx), :]
    return color


def draw_box2d(image, qs, color=(255,255,255), thickness=2):
    """Draw 2D box on image"""
    assert qs[0] >= 0
    assert qs[1] >= 0
    assert qs[2] < image.shape[1]
    assert qs[3] < image.shape[2]
    cv2.rectangle(image, (int(qs[0]),int(qs[1])), (int(qs[2]),int(qs[3])),
        color, thickness, cv2.LINE_AA)
    return image


def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    imsize = image.shape
    off = [False]*qs.shape[0]
    for i in range(qs.shape[0]):
        if (qs[i,0]<0) or (qs[i,0]>imsize[1]-1):
            if (qs[i,1]<0) or (qs[i,1]>imsize[0]-1):
                off[i] = True
        # qs[i,0] = min(max(qs[i,0], 0), imsize[1]-1)
        # qs[i,1] = min(max(qs[i,1], 0), imsize[0]-1)
    if sum(off) > 3:
        return image

    qs = qs.astype(np.int32)
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
    return image


def show_disparity(disparity, is_depth, extent=None):
    if is_depth:
        img = disparity
    else:
        img = Image.fromarray(disparity)

    plt.figure(figsize=[2*x for x in plt.rcParams["figure.figsize"]])
    plt.imshow(img, extent=extent, cmap='magma')
    plt.show()


def show_image(img, extent=None, axis=False, inline=True):
    if inline:
        pil_im = Image.fromarray(img)
        plt.figure(figsize=[2*x for x in plt.rcParams["figure.figsize"]])
        plt.imshow(pil_im, extent=extent)
        if not axis:
            plt.axis('off')
        plt.show()
    else:
        Image.fromarray(img).show()


def show_boxes_bev(boxes):
    raise NotImplementedError


def show_lidar_on_image(pc, img, boxes=None, show=True, inline=True, colormethod='depth', return_image=False):
    """Project LiDAR points to image"""
    import matplotlib.pyplot as plt
    img1 = np.copy(img.data)
    box2d_image = bbox.Box2D([0, 0, img1.shape[1], img1.shape[0]], img.calibration)
    points_in_view_filter = maskfilters.filter_points_in_image_frustum(pc, box2d_image, img.calibration)
    pc_proj_img = pc.filter(points_in_view_filter, inplace=False).project(img.calibration)

    for i in range(len(pc_proj_img)):
        if colormethod == 'depth':
            depth = pc_proj_img.depth[i]
            color = get_lidar_color(depth, mode='depth')
        elif 'channel' in colormethod:
            channel = int(colormethod.split('-')[-1])
            color = get_lidar_color(pc_proj_img.data[i,channel-2], mode='randint')
        else:
            raise NotImplementedError
        cv2.circle(img1, (int(np.round(pc_proj_img[i,0])),
                          int(np.round(pc_proj_img[i,1]))),
                   2, color=tuple(color), thickness=-1)
    if boxes is None:
        if show:
            show_image(img1, inline=inline)
        if return_image:
            return img1
    else:
        img1_n = deepcopy(img)
        img1_n.data = img1
        if show:
            show_image_with_boxes(img1_n, boxes, inline=inline)
        if return_image:
            return img1_n


def show_image_with_boxes(img, boxes, inline=False, box_colors='green',
        with_mask=False, show=True, return_images=False, addbox=[]):
    ''' Show image with bounding boxes '''
    img1 = np.copy(img.data)

    # Make appropriate types
    if isinstance(boxes, list):
        boxes = np.asarray(boxes)
    elif isinstance(boxes, (np.ndarray, DataContainer)):
        pass
    else:
        boxes = np.asarray([boxes])

    # Get color
    if box_colors is None:
        box_colors = ['green'] * len(boxes)
    elif isinstance(box_colors, str):
        box_colors = [box_colors] * len(boxes)

    # show each box
    mask = None
    for i, (box, col) in enumerate(zip(boxes, box_colors)):
        if isinstance(col, str):
            col = parse_color_string(col)

        # Show box
        if isinstance(box, Box2D) or (isinstance(box, (BoxDetection, MaskDetection)) and isinstance(box.box, Box2D)):
            if isinstance(box, (BoxDetection, MaskDetection)):
                if isinstance(box, MaskDetection):
                    mask = box.mask
                box = box.box
            cv2.rectangle(img1, (int(box.xmin),int(box.ymin)),
                (int(box.xmax),int(box.ymax)), col, 2)
        elif isinstance(box, (VehicleState, Box3D)) or (isinstance(box, (BoxDetection, MaskDetection)) and isinstance(box.box, Box3D)):
            if isinstance(box, VehicleState):
                box = box.box
            elif isinstance(box, (BoxDetection, MaskDetection)):
                if isinstance(box, MaskDetection):
                    mask = box.mask
                box = box.box
            if maskfilters.box_in_fov(box, img.calibration):
                corners_3d_in_image = box.project_corners_to_2d_image_plane(img.calibration, squeeze=True)
                img1 = draw_projected_box3d(img1, corners_3d_in_image, color=col)
        else:
            raise NotImplementedError(type(box))
        if addbox:
            cv2.rectangle(img1, (int(addbox[0]), int(addbox[1])), (int(addbox[2]), int(addbox[3])), (255,0,0), 2)

        # Show mask
        if (mask is not None) and (with_mask):
            mask_color = np.array([0,255,0], dtype='uint8')
            mask_img = np.where(mask.data[...,None], mask_color, img1)
            img1 = cv2.addWeighted(img1, 0.8, mask_img, 0.2, 0)

    # Plot results-----------------------
    if show:
        show_image(img1, inline=inline)
    if return_images:
        return img1


def show_objects_on_image(img, objects, projection='2d', **kwargs):
    objects = [obj for obj in objects if maskfilters.box_in_fov(obj.box3d, img.calibration)]
    if projection == '2d':
        boxes = [obj.box3d.project_to_2d_bbox(img.calibration) for obj in objects]
    elif projection == '3d':
        boxes = [obj.box3d for obj in objects]
    else:
        raise NotImplementedError(projection)
    return show_image_with_boxes(img, boxes, **kwargs)


def show_lidar_bev_with_boxes(point_cloud, boxes=[], extent=None, ground=None,
                   box_colors='white', filter_in_im=False, flipx=True, flipy=True, flipxy=True,
                   inline=True, lines=None, line_colors=None, bev_size=[500, 500],
                   colormethod='depth', show=True, return_image=False):
    """
    Show lidar and the detection results (optional) in BEV

    :point_cloud - lidar in the lidar frame
    """
    if type(boxes) == list:
        boxes = np.asarray(boxes)
    elif type(boxes) != np.ndarray:
        boxes = np.asarray([boxes])

    # Filter points
    if extent is not None:
        # Filter lidar outside extent
        point_filter = maskfilters.filter_points(point_cloud, extent, ground)
        pc2 = point_cloud[point_filter,:]

        # Filter labels outside extent
        for box in boxes:
            box.change_origin(NominalOriginStandard)
        box_filter = maskfilters.filter_boxes_extent(boxes, extent)
        boxes = boxes[box_filter]
        if type(box_colors) in [list, np.ndarray]:
            box_colors = box_colors[box_filter]
    else:
        pc2 = point_cloud.data

    # Get maxes and mins
    min_range = min(0, np.min(pc2[:,0]))
    max_range = max(min_range+10.0, np.max(pc2[:,0]))
    min_width = np.min(pc2[:,1])
    max_width = max(min_width+2.0, np.max(pc2[:,1]))

    boxes_show = []
    boxes_show_corners = []
    for i, box in enumerate(boxes):
        # Show box
        if isinstance(box, Box2D) or (isinstance(box, BoxDetection) and isinstance(box.box, Box2D)):
            continue  # cannot show 2D boxes
        elif isinstance(box, (VehicleState, Box3D)) or (isinstance(box, BoxDetection) and isinstance(box.box, Box3D)):
            if isinstance(box, VehicleState):
                box = box.box
            elif isinstance(box, BoxDetection):
                box = box.box
        else:
            raise NotImplementedError(type(box))

        # Corners in bev --  ***assumes for now pc z axis is up
        box.change_origin(point_cloud.calibration.origin)
        boxes_show.append(box)
        bev_corners = box.corners[:,:2]
        boxes_show_corners.append(bev_corners)

        # Update domain based on bbox
        min_range = min(min_range, min(bev_corners[:,0]) - 5)
        max_range = max(max_range, max(bev_corners[:,0]) + 5)
        min_width = min(min_width, min(bev_corners[:,1]) - 2)
        max_width = max(max_width, max(bev_corners[:,1]) + 2)

    # define the size of the image and scaling factor
    img1 = 0 * np.ones([bev_size[0], bev_size[1], 3], dtype=np.uint8)
    width_scale = (max_width - min_width) / bev_size[0]
    range_scale = (max_range - min_range) / bev_size[1]
    min_arr = np.array([min_range, min_width])
    sc_arr =  np.array([range_scale, width_scale])
    pc_bev = (pc2[:,[0,1]] - min_arr) / sc_arr

    # Colormap
    depth = np.linalg.norm(pc2[:,[0,1]], axis=1)

    # Make image by adding circles
    for i in range(pc_bev.shape[0]):
        if colormethod == 'depth':
            color = get_lidar_color(depth[i], mode='depth')
        elif colormethod == 'confidence':
            color = get_lidar_color(pc2[i,4], mode='confidence')
        else:
            raise NotImplementedError

        # Place in coordinates
        cv2.circle(img1, (int(np.round(pc_bev[i,0])),
            int(np.round(pc_bev[i,1]))),
            2, color=tuple(color), thickness=-1)

    # Add labels
    if type(box_colors) not in [list, np.ndarray]:
        ltmp = np.copy(box_colors)
        box_colors = [ltmp for _ in range(len(boxes_show))]
    for i, (box, bev_corners) in enumerate(zip(boxes_show, boxes_show_corners)):
        if isinstance(box_colors[i], (str, np.ndarray)):
            lcolor = parse_color_string(box_colors[i])
        else:
            assert isinstance(box_colors[i], tuple), f'{box_colors[i]}, {type(box_colors[i])}'
            lcolor = box_colors[i]
        box3d_pts_2d = (bev_corners - min_arr) / sc_arr
        img1 = draw_projected_box3d(img1, box3d_pts_2d, color=lcolor, thickness=2)

    # Add lines to the image if passed in
    if lines is not None:
        def plot_line(img1, line, line_color):
            """Assume line is a 2xn array"""
            color = parse_color_string(line_color)
            for p1, p2 in zip(line[:, :-1].T, line[:, 1:].T):
                p1_sc = tuple([int(p) for p in (p1 - min_arr)/sc_arr])
                p2_sc = tuple([int(p) for p in (p2 - min_arr)/sc_arr])
                cv2.line(img1, p1_sc, p2_sc, color, 5)

        if line_colors is None:
            line_colors = 'white'

        # If line is a list, it is a list of lines which are arrays
        if type(lines) is list:
            if type(line_colors) is not list:
                line_colors = [line_colors for _ in len(lines)]
            for l, lc in zip(lines, line_colors):
                plot_line(img1, l, lc)
        elif type(lines) is np.ndarray:
            plot_line(img1, lines, line_colors)
        else:
            raise RuntimeError('Unknown line type')

    viz_extent = [min_range, max_range, min_width, max_width]
    if flipx:
        img1 = np.flip(img1, axis=1)
        viz_extent = [viz_extent[1], viz_extent[0], viz_extent[2], viz_extent[3]]
    if flipy:
        img1 = np.flip(img1, axis=0)
        viz_extent = [viz_extent[0], viz_extent[1], viz_extent[3], viz_extent[2]]
    if flipxy:
        img1 = img1.transpose(1,0,2)
        viz_extent = [viz_extent[2], viz_extent[3], viz_extent[0], viz_extent[1]]

    if show:
        show_image(img1, extent=viz_extent, inline=inline)
    if return_image:
        return img1
