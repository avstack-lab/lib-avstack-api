
import cv2
import matplotlib.pyplot as plt
import numpy as np


lidar_cmap = plt.cm.get_cmap("hsv", 256)
lidar_cmap = np.array([lidar_cmap(i) for i in range(256)])[:, :3] * 255


def get_lidar_color(value, mode="depth"):
    # Min range = 0 --> 1
    # Max range = 100 --> 255
    if mode == "depth":
        scaling = 100
    elif mode == "confidence":
        scaling = 2
    elif mode == "randint":
        scaling = 50
    else:
        raise NotImplementedError(mode)

    idx = max(1, min(255, 255 * value / scaling))
    color = lidar_cmap[int(idx), :]
    return color


def draw_box2d(image, qs, color=(255, 255, 255), thickness=2):
    """Draw 2D box on image"""
    assert qs[0] >= 0
    assert qs[1] >= 0
    assert qs[2] < image.shape[1]
    assert qs[3] < image.shape[2]
    cv2.rectangle(
        image,
        (int(qs[0]), int(qs[1])),
        (int(qs[2]), int(qs[3])),
        color,
        thickness,
        cv2.LINE_AA,
    )
    return image


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2, ID=None, fontscale=1):
    """Draw 3d bounding box in image
    qs: (8,3) array of vertices for the 3d box in following order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    """
    imsize = image.shape
    off = [False] * qs.shape[0]
    for i in range(qs.shape[0]):
        if (qs[i, 0] < 0) or (qs[i, 0] > imsize[1] - 1):
            if (qs[i, 1] < 0) or (qs[i, 1] > imsize[0] - 1):
                off[i] = True
        # qs[i,0] = min(max(qs[i,0], 0), imsize[1]-1)
        # qs[i,1] = min(max(qs[i,1], 0), imsize[0]-1)
    if sum(off) > 3:
        return image

    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(
            image,
            (qs[i, 0], qs[i, 1]),
            (qs[j, 0], qs[j, 1]),
            color,
            thickness,
            cv2.LINE_AA,
        )

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(
            image,
            (qs[i, 0], qs[i, 1]),
            (qs[j, 0], qs[j, 1]),
            color,
            thickness,
            cv2.LINE_AA,
        )

        i, j = k, k + 4
        cv2.line(
            image,
            (qs[i, 0], qs[i, 1]),
            (qs[j, 0], qs[j, 1]),
            color,
            thickness,
            cv2.LINE_AA,
        )

    # name on top of box
    if ID is not None:
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        edge                   = 15
        sep                    = 4
        bottomLeftCornerOfText = (max(edge, np.min(qs[:,0])-sep)), max(edge, np.min(qs[:,1])-sep)
        fontColor              = (255,255,255)
        font_thickness         = 1
        lineType               = 2
        cv2.putText(image, str(ID), 
            bottomLeftCornerOfText, 
            font, 
            fontscale,
            fontColor,
            font_thickness,
            lineType)
        
    return image
