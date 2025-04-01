import avapi.carla
import avapi.evaluation
import avapi.kitti
import avapi.mot15
import avapi.nuimages
import avapi.nuscenes
import avapi.opv2v
import avapi.ugv
import avapi.visualize


def get_scene_manager(dataset, data_dir, split, verbose=False):
    ds = dataset.lower()
    if ds == "mot15":
        SM = avapi.mot15.MOT15SceneManager
    elif ds == "kitti":
        SM = avapi.kitti.KittiScenesManager
    elif ds == "nuscenes":
        SM = avapi.nuscenes.nuScenesManager
    elif ds == "carla":
        SM = avapi.carla.CarlaScenesManager
    else:
        raise NotImplementedError
    return SM(data_dir, split, verbose=verbose)
