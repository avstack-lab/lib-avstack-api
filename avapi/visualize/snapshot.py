from copy import deepcopy

import avstack
import cv2
import numpy as np
from avstack import maskfilters
from avstack.datastructs import DataContainer
from avstack.environment.objects import ObjectState
from avstack.geometry import Box2D, Box3D, bbox
from avstack.geometry.transformations import project_to_image
from avstack.modules.perception.detections import BoxDetection, MaskDetection
from avstack.modules.tracking.tracks import BasicBoxTrack2D, BasicBoxTrack3D, GroupTrack
from PIL import Image

from avapi.utils import parse_color_string

from .base import draw_projected_box3d, get_lidar_color


def show_disparity(disparity, is_depth, extent=None):
    import matplotlib.pyplot as plt

    if is_depth:
        img = disparity
    else:
        img = Image.fromarray(disparity)

    plt.figure(figsize=[2 * x for x in plt.rcParams["figure.figsize"]])
    plt.imshow(img, extent=extent, cmap="magma")
    plt.show()


def show_image(img, extent=None, axis=False, inline=True, grayscale=False):
    import matplotlib.pyplot as plt

    if inline:
        pil_im = Image.fromarray(img)
        plt.figure(figsize=[2 * x for x in plt.rcParams["figure.figsize"]])
        img_plot = plt.imshow(
            pil_im, extent=extent, cmap=("gray" if grayscale else None)
        )
        if not axis:
            plt.axis("off")
        plt.show()
    else:
        img_plot = Image.fromarray(img).show()
    return img_plot


def show_boxes_bev(
    boxes,
    vectors=[],
    extent=None,
    ground=None,
    box_colors="white",
    filter_in_im=False,
    flipx=True,
    flipy=True,
    flipxy=True,
    inline=True,
    lines=None,
    line_colors=None,
    bev_size=[500, 500],
    colormethod="depth",
    show=True,
    return_image=False,
):
    min_range, max_range = 0, 0
    min_width, max_width = 0, 0

    boxes_show = []
    boxes_show_corners = []
    for i, box in enumerate(boxes):
        # Show box
        if isinstance(box, Box2D) or (
            isinstance(box, BoxDetection) and isinstance(box.box, Box2D)
        ):
            continue  # cannot show 2D boxes
        elif (
            isinstance(box, (ObjectState, Box3D))
            or (isinstance(box, BoxDetection) and isinstance(box.box, Box3D))
            or (isinstance(box, BasicBoxTrack3D))
            or (isinstance(box, GroupTrack) and isinstance(box.state, BasicBoxTrack3D))
        ):
            if isinstance(box, (BoxDetection, BasicBoxTrack3D, ObjectState)):
                box = box.box
            elif isinstance(box, GroupTrack):
                box = box.state.box
        else:
            raise NotImplementedError(type(box))
        boxes_show.append(box)
        bev_corners = box.corners[:, :2]
        boxes_show_corners.append(bev_corners)

        # Update domain based on bbox
        min_range = min(min_range, min(bev_corners[:, 0]) - 5)
        max_range = max(max_range, max(bev_corners[:, 0]) + 5)
        min_width = min(min_width, min(bev_corners[:, 1]) - 2)
        max_width = max(max_width, max(bev_corners[:, 1]) + 2)

    # define the size of the image and scaling factor
    img1 = 0 * np.ones([bev_size[0], bev_size[1], 3], dtype=np.uint8)
    if extent is None:
        width_scale = (max_width - min_width) / bev_size[0]
        range_scale = (max_range - min_range) / bev_size[1]
        min_arr = np.array([min_range, min_width])
    else:
        width_scale = (extent[1][1] - extent[1][0]) / bev_size[0]
        range_scale = (extent[0][1] - extent[0][0]) / bev_size[1]
        min_arr = np.array([extent[0][0], extent[1][0]])
    sc_arr = np.array([range_scale, width_scale])

    # Add labels
    if type(box_colors) not in [list, np.ndarray]:
        ltmp = np.copy(box_colors)
        box_colors = [ltmp for _ in range(len(boxes_show))]
    for i, (box, bev_corners) in enumerate(zip(boxes_show, boxes_show_corners)):
        if isinstance(box_colors[i], (str, np.ndarray)):
            lcolor = parse_color_string(box_colors[i])
        else:
            assert isinstance(
                box_colors[i], tuple
            ), f"{box_colors[i]}, {type(box_colors[i])}"
            lcolor = box_colors[i]
        box3d_pts_2d = (bev_corners - min_arr) / sc_arr
        img1 = draw_projected_box3d(img1, box3d_pts_2d, color=lcolor, thickness=2)

    # Add tracks
    for i, vec in enumerate(vectors):
        head = (vec.head.x[:2] - min_arr) / sc_arr
        tail = (vec.tail.x[:2] - min_arr) / sc_arr
        color = (0, 255, 0)
        thickness = 4
        img1 = cv2.arrowedLine(
            img1, tuple(map(int, head)), tuple(map(int, tail)), color, thickness
        )

    # Add lines to the image if passed in
    if lines is not None:

        def plot_line(img1, line, line_color):
            """Assume line is a 2xn array"""
            color = parse_color_string(line_color)
            for p1, p2 in zip(line[:, :-1].T, line[:, 1:].T):
                p1_sc = tuple([int(p) for p in (p1 - min_arr) / sc_arr])
                p2_sc = tuple([int(p) for p in (p2 - min_arr) / sc_arr])
                cv2.line(img1, p1_sc, p2_sc, color, 5)

        if line_colors is None:
            line_colors = "white"

        # If line is a list, it is a list of lines which are arrays
        if type(lines) is list:
            if type(line_colors) is not list:
                line_colors = [line_colors for _ in len(lines)]
            for l, lc in zip(lines, line_colors):
                plot_line(img1, l, lc)
        elif type(lines) is np.ndarray:
            plot_line(img1, lines, line_colors)
        else:
            raise RuntimeError("Unknown line type")

    if extent is None:
        viz_extent = [min_range, max_range, min_width, max_width]
    else:
        viz_extent = [*extent[0], *extent[1]]

    if flipx:
        img1 = np.flip(img1, axis=1)
        viz_extent = [viz_extent[1], viz_extent[0], viz_extent[2], viz_extent[3]]
    if flipy:
        img1 = np.flip(img1, axis=0)
        viz_extent = [viz_extent[0], viz_extent[1], viz_extent[3], viz_extent[2]]
    if flipxy:
        img1 = img1.transpose(1, 0, 2)
        viz_extent = [viz_extent[2], viz_extent[3], viz_extent[0], viz_extent[1]]

    if show:
        show_image(img1, extent=viz_extent, inline=inline)
    if return_image:
        return img1


def show_lidar_on_image(
    pc, img, boxes=None, show=True, inline=True, colormethod="depth", return_image=False
):
    """Project LiDAR points to image"""

    img1 = np.copy(img.data)
    if img.calibration.channel_order == "bgr":
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    box2d_image = bbox.Box2D([0, 0, img1.shape[1], img1.shape[0]], img.calibration)
    points_in_view_filter = maskfilters.filter_points_in_image_frustum(
        pc, box2d_image, img.calibration
    )
    pc_proj_img = pc.filter(points_in_view_filter, inplace=False).project(
        img.calibration
    )

    # get colors for lidar pcs
    if colormethod == "depth":
        depths = pc_proj_img.depth
        pt_colors = get_lidar_color(depths, mode="depth")
    elif "channel" in colormethod:
        raise NotImplementedError
        # channel = int(colormethod.replace('channel', ''))
        # pt_colors = get_lidar_color(pc2[:, channel], mode="channel")
        # channel = int(colormethod.split("-")[-1])
        # pt_colors = get_lidar_color(pc_proj_img.data[:, channel - 2], mode="randint")
    else:
        raise NotImplementedError

    # add each point to image
    for i in range(len(pc_proj_img)):
        cv2.circle(
            img1,
            # (int(np.round(pc_proj_img[i, 0])), int(np.round(pc_proj_img[i, 1]))),
            (int(pc_proj_img[i, 0]), int(pc_proj_img[i, 1])),
            2,
            color=tuple(pt_colors[i]),
            thickness=-1,
        )
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


def show_image_with_boxes(
    img,
    boxes,
    text=None,
    inline=False,
    box_colors="green",
    box_thickness=3,
    with_mask=False,
    show_IDs=True,
    fontscale=1,
    font_thickness=3,
    show=True,
    return_image=False,
    addbox=[],
):
    """Show image with bounding boxes"""
    try:
        img1 = np.copy(img.rgb_image)
    except AttributeError:
        img1 = np.copy(img.grayscale_image)
    # if img.calibration.channel_order == "bgr":
    #     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # Make appropriate types
    if isinstance(boxes, list):
        boxes = np.asarray(boxes)
    elif isinstance(boxes, (np.ndarray, DataContainer)):
        pass
    else:
        boxes = np.asarray([boxes])

    # Get color
    if box_colors is None:
        box_colors = ["green"] * len(boxes)
    elif isinstance(box_colors, str):
        box_colors = [box_colors] * len(boxes)
    elif (
        isinstance(box_colors, (tuple, list))
        and len(box_colors) == 3
        and isinstance(box_colors[0], int)
    ):
        box_colors = [box_colors] * len(boxes)

    # Get IDs
    if show_IDs:
        box_IDs = [box.ID for box in boxes]
    else:
        box_IDs = [None] * len(boxes)

    # Show each box
    mask = None
    for i, (box, col, ID) in enumerate(zip(boxes, box_colors, box_IDs)):
        if isinstance(col, str):
            col = parse_color_string(col)

        # Show box
        if isinstance(box, Box2D) or (
            isinstance(box, (ObjectState, BoxDetection, MaskDetection, BasicBoxTrack2D))
            and isinstance(box.box, Box2D)
        ):
            if isinstance(
                box, (ObjectState, BoxDetection, MaskDetection, BasicBoxTrack2D)
            ):
                if isinstance(box, MaskDetection):
                    mask = box.mask
                box = box.box
            # add box
            cv2.rectangle(
                img1,
                (int(box.xmin), int(box.ymin)),
                (int(box.xmax), int(box.ymax)),
                col,
                box_thickness,
            )
            bl_edge = (box.xmin, box.ymin)
            # add text
            if text is not None:
                add_text_to_image(
                    img1,
                    bl_edge,
                    text[i],
                    fontscale=fontscale,
                    font_thickness=font_thickness,
                )
            # add id
            add_text_to_image(
                img1, bl_edge, ID, fontscale=fontscale, font_thickness=font_thickness
            )
        elif (
            isinstance(box, (Box3D, BasicBoxTrack3D))
            or (
                isinstance(box, (ObjectState, BoxDetection, MaskDetection))
                and isinstance(box.box, Box3D)
            )
            or (isinstance(box, GroupTrack) and isinstance(box.state, BasicBoxTrack3D))
        ):
            if isinstance(box, (ObjectState, BasicBoxTrack3D)):
                box = box.box
            elif isinstance(box, (BoxDetection, MaskDetection)):
                if isinstance(box, MaskDetection):
                    mask = box.mask
                box = box.box
            elif isinstance(box, GroupTrack):
                box = box.state.box
            if maskfilters.box_in_fov(box, img.calibration):
                corners_3d_in_image = box.project_corners_to_2d_image_plane(
                    img.calibration
                )
                img1 = draw_projected_box3d(
                    img1,
                    corners_3d_in_image,
                    thickness=box_thickness,
                    color=col,
                    ID=ID,
                    fontscale=fontscale,
                )
        elif isinstance(
            box,
            (
                avstack.modules.tracking.tracks.XyzFromRazelTrack,
                avstack.modules.perception.detections.RazelDetection,
            ),
        ):
            pts_box = np.array([[-box.x[1], -box.x[2], box.x[0]]])
            pt = project_to_image(pts_box, img.calibration.P)[0]
            radius = 6
            cv2.circle(
                img1, (int(pt[0]), int(pt[1])), radius, color=(0, 255, 0), thickness=-1
            )
            bl_edge = (pt[0], pt[1])
            add_text_to_image(img1, bl_edge, ID, fontscale=fontscale)
        else:
            raise NotImplementedError(type(box))
        if addbox:
            cv2.rectangle(
                img1,
                (int(addbox[0]), int(addbox[1])),
                (int(addbox[2]), int(addbox[3])),
                (255, 0, 0),
                box_thickness,
            )

        # Show mask
        if (mask is not None) and (with_mask):
            if (len(img.shape) == 3) and (img.shape[2] == 3):
                mask_color = np.array([0, 255, 0], dtype="uint8")
                mask_img = np.where(mask.data[..., None], mask_color, img1)
                img1 = cv2.addWeighted(img1, 0.7, mask_img, 0.3, 0)
            else:
                mask_color = np.array([255], dtype="uint8")
                mask_img = np.where(mask.data, mask_color, img1)
                img1 = cv2.addWeighted(img1, 1.0, mask_img, 0, 0)

    # Plot results-----------------------
    if show:
        show_image(
            img1, inline=inline, grayscale=(len(img.shape) < 3 or img.shape[2] == 1)
        )
    if return_image:
        return img1


def add_text_to_image(img, bl_edge, text, fontscale=1, font_thickness=3):
    if text is not None:
        # name on top of box
        font = cv2.FONT_HERSHEY_SIMPLEX
        edge = 15
        sep = 4
        bottomLeftCornerOfText = (int(max(edge, bl_edge[0] - sep))), int(
            max(edge, bl_edge[1] - sep)
        )
        fontColor = (255, 255, 255)
        lineType = 2
        x, y = bottomLeftCornerOfText
        dy = 10
        for line in str(text).splitlines():
            cv2.putText(
                img,
                line,
                (x, y),
                font,
                fontscale,
                fontColor,
                font_thickness,
                lineType,
            )
            y += 20


def show_lidar_bev_with_boxes(
    pc,
    boxes=[],
    vectors=[],
    extent=None,
    ground=None,
    box_colors="white",
    box_filled=False,
    box_thickness=3,
    filter_in_im=False,
    flipx=True,
    flipy=True,
    flipxy=True,
    inline=True,
    lines=None,
    line_colors=None,
    bev_size=[500, 500],
    fov=None,
    fov_color: str = "#069Af3",
    fov_filled: bool = True,
    fov_filled_alpha: float = 0.1,
    colormethod: str = "depth",
    background_color: str = "black",
    rescale: bool = True,
    show: bool = True,
    return_image: bool = False,
    scale_return_image: bool = False,
):
    """
    Show lidar and the detection results (optional) in BEV

    :pc - lidar in the lidar frame
    :extent -  3D area in the form
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]]

    """
    # Make appropriate types
    if isinstance(boxes, list):
        boxes = np.asarray(boxes)
    elif isinstance(boxes, (np.ndarray, DataContainer)):
        pass
    else:
        boxes = np.asarray([boxes])
    boxes = np.array(
        [box.change_reference(pc.calibration.reference, inplace=False) for box in boxes]
    )

    # Filter points
    if extent is not None:
        # Filter lidar outside extent
        point_filter = maskfilters.filter_points(pc, extent, ground)
        pc2 = pc[point_filter, :]

        # Filter labels outside extent
        box_filter = maskfilters.filter_boxes_extent(boxes, extent)
        boxes = boxes[box_filter]
        if type(box_colors) in [list, np.ndarray]:
            box_colors = [col for col, yesno in zip(box_colors, box_filter) if yesno]
    else:
        pc2 = pc.data

    # update extent
    if extent is not None:
        extent_max = [
            (min(pc[:, 0]), max(pc[:, 1])),
            (min(pc[:, 1]), max(pc[:, 1])),
            (min(pc[:, 2]), max(pc[:, 2])),
        ]
        extent_use = []
        for ex, ex_max in zip(extent, extent_max):
            ex_in = [e1 if e1 is not None else em for e1, em in zip(ex, ex_max)]
            extent_use.append(tuple(ex_in))
    else:
        extent_use = extent

    # Get maxes and mins
    if rescale:
        if pc2.shape[0] > 0:
            min_range = min(0, np.min(pc2[:, 0]))
            max_range = max(min_range + 10.0, np.max(pc2[:, 0]))
            min_width = np.min(pc2[:, 1])
            max_width = max(min_width + 2.0, np.max(pc2[:, 1]))
        else:
            min_range = 0
            max_range = 0
            min_width = 0
            max_width = 0
    else:
        assert extent_use is not None
        min_range, max_range = extent_use[0]
        min_width, max_width = extent_use[1]

    boxes_show = []
    boxes_show_corners = []
    for i, box in enumerate(boxes):
        # Show box
        if isinstance(box, Box2D) or (
            isinstance(box, BoxDetection) and isinstance(box.box, Box2D)
        ):
            continue  # cannot show 2D boxes
        elif (
            isinstance(box, (ObjectState, Box3D))
            or (isinstance(box, BoxDetection) and isinstance(box.box, Box3D))
            or (isinstance(box, BasicBoxTrack3D))
            or (isinstance(box, GroupTrack) and isinstance(box.state, BasicBoxTrack3D))
        ):
            if isinstance(box, (BoxDetection, BasicBoxTrack3D, ObjectState)):
                box = box.box
            elif isinstance(box, GroupTrack):
                box = box.state.box
        else:
            raise NotImplementedError(type(box))

        # Corners in bev --  ***assumes for now pc z axis is up
        # box.change_reference(pc.calibration.reference, inplace=False)
        boxes_show.append(box)
        bev_corners = box.corners[:, :2]
        boxes_show_corners.append(bev_corners)

        # Update domain based on bbox
        min_range = min(min_range, min(bev_corners[:, 0]) - 5)
        max_range = max(max_range, max(bev_corners[:, 0]) + 5)
        min_width = min(min_width, min(bev_corners[:, 1]) - 2)
        max_width = max(max_width, max(bev_corners[:, 1]) + 2)

    # update extent with new ranges/widths
    if extent_use is not None:
        extent_use[0] = (min(min_range, extent_use[0][0]), max(max_range, extent_use[0][1]))
        extent_use[1] = (min(min_width, extent_use[1][0]), max(max_width, extent_use[1][1]))

    # define the size of the image and scaling factor
    if background_color == "black":
        img1 = 0 * np.ones([bev_size[0], bev_size[1], 3], dtype=np.uint8)
    elif background_color == "white":
        img1 = 255 * np.ones([bev_size[0], bev_size[1], 3], dtype=np.uint8)
    else:
        raise NotImplementedError(background_color)

    # get scaling
    if extent_use is None:
        width_scale = (max_width - min_width) / bev_size[0]
        range_scale = (max_range - min_range) / bev_size[1]
        min_arr = np.array([min_range, min_width])
    else:
        width_scale = (extent_use[1][1] - extent_use[1][0]) / bev_size[0]
        range_scale = (extent_use[0][1] - extent_use[0][0]) / bev_size[1]
        min_arr = np.array([extent_use[0][0], extent_use[1][0]])
    sc_arr = np.array([range_scale, width_scale])
    pc_bev = (pc2[:, [0, 1]] - min_arr) / sc_arr

    # add the field of view by blending
    if fov is not None:
        # add fov boundary without alpha
        boundary_bev = ((fov.boundary[:, [0, 1]] - min_arr) / sc_arr).astype(int)
        boundary_bev = boundary_bev.reshape((-1, 1, 2))
        thickness = 3
        cv2.polylines(
            img=img1,
            pts=[boundary_bev],
            color=parse_color_string(fov_color),
            thickness=thickness,
            isClosed=True,
        )
        # add fov filled with alpha
        if fov_filled:
            img2 = img1.copy()
            cv2.fillPoly(
                img=img2,
                pts=[boundary_bev],
                color=parse_color_string(fov_color),
            )
        w = fov_filled_alpha
        img1 = cv2.addWeighted(img1, 1 - w, img2, w, 0)

    # get colors for lidar pcs
    if colormethod == "depth":
        depths = np.linalg.norm(pc2[:, [0, 1]], axis=1)
        pt_colors = get_lidar_color(depths, mode="depth")
    elif colormethod == "confidence":
        pt_colors = get_lidar_color(pc2[:, 4], mode="confidence")
    elif "channel" in colormethod:
        channel = int(colormethod.split("-")[1])
        pt_colors = get_lidar_color(pc2[:, channel], mode="channel")
    elif colormethod == "black":
        pt_colors = 0 * np.ones((len(pc2), 3), dtype=float)
    elif colormethod == "white":
        pt_colors = 255 * np.ones((len(pc2), 3), dtype=float)
    else:
        raise NotImplementedError

    # Make image by adding circles
    for i in range(pc_bev.shape[0]):
        # Place in coordinates
        cv2.circle(
            img1,
            # (int(np.round(pc_bev[i, 0])), int(np.round(pc_bev[i, 1]))),
            (int(pc_bev[i, 0]), int(pc_bev[i, 1])),
            2,
            color=tuple(pt_colors[i]),
            thickness=-1,
        )

    # Add labels
    if type(box_colors) not in [list, np.ndarray]:
        ltmp = np.copy(box_colors)
        box_colors = [ltmp for _ in range(len(boxes_show))]
    for i, (box, bev_corners) in enumerate(zip(boxes_show, boxes_show_corners)):
        if isinstance(box_colors[i], (str, np.ndarray)):
            lcolor = parse_color_string(box_colors[i])
        else:
            assert isinstance(
                box_colors[i], tuple
            ), f"{box_colors[i]}, {type(box_colors[i])}"
            lcolor = box_colors[i]
        box3d_pts_2d = (bev_corners - min_arr) / sc_arr
        img1 = draw_projected_box3d(
            img1, box3d_pts_2d, color=lcolor, thickness=box_thickness, filled=box_filled
        )

    # Add tracks
    for i, vec in enumerate(vectors):
        vec.change_reference(pc.calibration.reference, inplace=True)
        head = (vec.head.x[:2] - min_arr) / sc_arr
        tail = (vec.tail.x[:2] - min_arr) / sc_arr
        color = (0, 255, 0)
        thickness = 4
        img1 = cv2.arrowedLine(
            img1, tuple(map(int, head)), tuple(map(int, tail)), color, thickness
        )

    # Add lines to the image if passed in
    if lines is not None:

        def plot_line(img1, line, line_color):
            """Assume line is a 2xn array"""
            color = parse_color_string(line_color)
            for p1, p2 in zip(line[:, :-1].T, line[:, 1:].T):
                p1_sc = tuple([int(p) for p in (p1 - min_arr) / sc_arr])
                p2_sc = tuple([int(p) for p in (p2 - min_arr) / sc_arr])
                cv2.line(img1, p1_sc, p2_sc, color, 5)

        if line_colors is None:
            line_colors = "white"

        # If line is a list, it is a list of lines which are arrays
        if type(lines) is list:
            if type(line_colors) is not list:
                line_colors = [line_colors for _ in len(lines)]
            for l, lc in zip(lines, line_colors):
                plot_line(img1, l, lc)
        elif type(lines) is np.ndarray:
            plot_line(img1, lines, line_colors)
        else:
            raise RuntimeError("Unknown line type")

    nominal_extent = [min_range, max_range, min_width, max_width]
    if extent_use is None:
        viz_extent = nominal_extent
    else:
        viz_extent = [*extent_use[0], *extent_use[1]]

    if flipx:
        img1 = np.flip(img1, axis=1)
        viz_extent = [viz_extent[1], viz_extent[0], viz_extent[2], viz_extent[3]]
    if flipy:
        img1 = np.flip(img1, axis=0)
        viz_extent = [viz_extent[0], viz_extent[1], viz_extent[3], viz_extent[2]]
    if flipxy:
        img1 = img1.transpose(1, 0, 2)
        viz_extent = [viz_extent[2], viz_extent[3], viz_extent[0], viz_extent[1]]

    if show:
        img_plot = show_image(img1, extent=viz_extent, inline=inline)
        image_array = img_plot.get_array()

    if return_image:
        if scale_return_image:
            scale_ratio = width_scale / range_scale
            dx = bev_size[0] * scale_ratio if scale_ratio > 1 else bev_size[0]
            dy = bev_size[1] / scale_ratio if scale_ratio < 1 else bev_size[1]
            new_size = (int(dx), int(dy))
            return cv2.resize(img1, new_size)
        else:
            return img1
