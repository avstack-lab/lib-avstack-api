from functools import partial
from multiprocessing import Pool

import cv2
import ipywidgets as wg
from IPython.display import display
from tqdm import tqdm

from avapi.visualize.snapshot import (
    show_boxes_bev,
    show_image_with_boxes,
    show_lidar_bev_with_boxes,
)


def make_movie_from_DM(
    DM,
    dataset_name,
    boxes=[],
    CAM="main_camera",
    save=False,
    show_in_notebook=True,
    *args,
    **kwargs,
):
    if DM.frames is None:
        print("NO FRAMES IN SCENE")
        return

    if dataset_name == "carla":
        i_start = 4
        i_end = len(DM.frames) - 5
    else:
        i_start = 0
        i_end = len(DM.frames)

    imgs = []
    boxes_out = []
    for frame_idx in range(i_start, i_end, 1):
        frame = DM.get_frames(sensor=CAM)[frame_idx]
        if len(boxes) == 0:
            img_boxes = DM.get_objects(frame, sensor=CAM)  # using ground truth
        else:
            img_boxes = boxes[frame_idx]
        img = DM.get_image(frame, sensor=CAM)

        boxes_out.append(img_boxes)
        imgs.append(img)

    make_movie(
        imgs,
        boxes_out,
        fps=DM.framerate,
        name=dataset_name,
        save=save,
        show_in_notebook=show_in_notebook,
        *args,
        **kwargs,
    )


def _get_image_with_box(projection, extent, img, pc, boxes, *args, **kwargs):
    from avapi.evaluation import ResultManager

    if projection == "img":
        if isinstance(boxes, ResultManager):
            img_out = boxes.visualize(
                image=img,
                projection="img",
                show=False,
                return_image=True,
                *args,
                **kwargs,
            )
        else:
            img_out = show_image_with_boxes(
                img, boxes, show=False, return_image=True, *args, **kwargs
            )
    elif projection == "bev":
        if pc is None:
            img_out = show_boxes_bev(
                boxes, extent=extent, show=False, return_image=True, *args, **kwargs
            )
        else:
            img_out = show_lidar_bev_with_boxes(
                pc, boxes, extent=extent, show=False, return_image=True, *args, **kwargs
            )
    else:
        raise NotImplementedError(projection)
    return img_out


def make_movie(
    raw_imgs,
    boxes=[],
    raw_pcs=None,
    fps=10,
    name="untitled",
    projection="img",
    save=False,
    show_in_notebook=True,
    extent=None,
    with_multi=False,
    nproc=5,
    suffix="scene_movie.mp4",
    *args,
    **kwargs,
):
    if len(boxes) == 0:
        boxes = [[]] * len(raw_imgs)
    if raw_pcs is None:
        raw_pcs = [None] * len(boxes)

    # process images (adding boxes to raw images)
    if with_multi:
        part_func = partial(_get_image_with_box, projection, extent)
        with Pool(nproc) as p:
            processed_imgs = list(
                tqdm(
                    p.istarmap(
                        part_func,
                        zip(raw_imgs, boxes),
                    ),
                    position=0,
                    leave=True,
                    total=len(boxes),
                )
            )
    else:
        processed_imgs = [
            _get_image_with_box(
                projection, extent, img=img, pc=pc, boxes=box, *args, **kwargs
            )
            for img, pc, box in tqdm(zip(raw_imgs, raw_pcs, boxes), total=len(boxes))
        ]
    print("done")
    height, width, layers = processed_imgs[0].shape
    size = (width, height)
    # generate movie
    if save:
        if suffix:
            movie_name = name + "_" + suffix
        else:
            movie_name = name

        video = cv2.VideoWriter(movie_name, cv2.VideoWriter_fourcc(*"DIVX"), fps, size)
        print("Saving movie")
        for img in processed_imgs:
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("done")

        # Deallocating memories taken for window creation
        cv2.destroyAllWindows()
        print("Scene video saved sucessfully")

        video.release()  # releasing the video generated

    # show in notebook
    if show_in_notebook:
        make_slider_view(processed_imgs)


def make_slider_view(imgs):
    import matplotlib.pyplot as plt

    def f(idx):
        axs_slider.imshow(imgs[idx])
        axs_slider.set_title("Frame %03i" % int(idx))
        fig.canvas.draw()
        display(fig)

    # make slider view
    fig, axs_slider = plt.subplots(1, 1)
    plt.axis("off")
    wg.interact(f, idx=wg.IntSlider(min=0, max=len(imgs) - 1, step=1, value=0))
