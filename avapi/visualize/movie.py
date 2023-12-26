import os
import avstack
import avapi
import cv2  
from tqdm import tqdm
import ipywidgets as wg
from IPython.display import SVG
from IPython.display import Video
from IPython.display import display


def make_movie_from_DM(DM, dataset_name, CAM='main_camera', save=False, show_in_notebook=True):
    if DM.frames is None:
        print("NO FRAMES IN SCENE")
        return 
            
    if dataset_name == 'carla':
        i_start = 4
        i_end = len(DM.frames) - 5
    else:
        i_start = 0
        i_end = len(DM.frames)
        
    imgs = []
    boxes = []
    for frame_idx in range(i_start, i_end, 1):
        frame = DM.get_frames(sensor=CAM)[frame_idx]
        img_boxes = DM.get_objects(frame, sensor=CAM) # using ground truth
        img = DM.get_image(frame, sensor=CAM)
        
        boxes.append(img_boxes)
        imgs.append(img)

    make_movie(imgs, boxes, fps=DM.framerate, name=dataset_name, save=save, show_in_notebook=show_in_notebook)


def make_movie(raw_imgs, boxes, fps=10, name="untitled", save=False, show_in_notebook=True):
    
    # process images (adding boxes to raw images)
    processed_imgs = []
    print('Processing images and boxes')
    for img, box in tqdm(zip(raw_imgs, boxes), total=len(boxes)):
        processed_imgs.append(avapi.visualize.snapshot.show_image_with_boxes(img, box, show=False, return_images=True))
    print("done")
    height, width, layers = processed_imgs[0].shape
    size = (width,height)
    # generate movie
    if save:
        movie_name = name + '_scene_movie.mp4'
    
        video = cv2.VideoWriter(movie_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
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
    fig, axs_slider = plt.subplots(1,1)
    plt.axis('off')
    wg.interact(f,idx=wg.IntSlider(min=0, max=len(imgs) - 1, step=1, value=0))





    

