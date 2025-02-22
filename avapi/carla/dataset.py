import glob
import json
import os
from typing import Tuple, Union

import numpy as np
from avstack import calibration
from avstack.config import DATASETS
from avstack.datastructs import DataContainer, DataContainerDecoder
from avstack.environment import ObjectStateDecoder
from avstack.geometry import GlobalOrigin3D, ReferenceFrame
from cv2 import imread
from tqdm import tqdm

from .._dataset import BaseSceneDataset, BaseSceneManager


def check_xor_for_none(a, b):
    if (a is None) and (b is None):
        raise ValueError("Both inputs cannot be none")
    if (a is not None) and (b is not None):
        raise ValueError("At least one input must be none")


def get_splits_scenes(data_dir, seed=1, frac_train=0.7, frac_val=0.3):
    CSM = CarlaScenesManager(data_dir)
    return CSM.make_splits_scenes(seed=seed, frac_train=frac_train, frac_val=frac_val)


def run_dataset_postprocessing(data_dir):
    """Run postprocessing to optimize dataset for usage

    Includes:
        - saving object representations in sensor frames
    """
    CSM = CarlaScenesManager(data_dir=data_dir)
    obj_dir = os.path.join(CSM.data_dir, "objects_sensor")
    for i_scene, CDM in enumerate(CSM):
        print("Running scene {} of {}".format(i_scene, len(CSM)))
        for frame in tqdm(CDM.frames):
            objects_global = CDM.get_objects_global(frame=frame)
            for sensor in CDM.sensor_IDs:
                calib = CDM.get_calibration(frame=frame, sensor=sensor)
                object_string = "\n".join(
                    [
                        obj.change_reference(calib.reference, inplace=False).encode()
                        for obj in objects_global
                    ]
                )
                global_file = CSM.get_object_file(
                    frame=frame, timestamp=None, is_agent=False, is_global=True
                )
                file = os.path.join(obj_dir, sensor, global_file.split("/")[-1])
                os.makedirs(os.path.dirname(file), exist_ok=True)
                with open(file, "w") as f:
                    f.write(object_string)


# _nominal_whitelist_types = ['car', 'pedestrian', 'bicycle',
#         'truck', 'motorcycle']
_nominal_whitelist_types = ["car", "bicycle", "truck", "motorcycle"]
_nominal_ignore_types = []


@DATASETS.register_module()
class CarlaScenesManager(BaseSceneManager):
    name = "CARLA"
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(
        self,
        data_dir,
        verbose=False,
        split_fracs={"train": 0.6, "val": 0.2, "test": 0.2},
        seed: int = 1,
    ):
        """
        data_dir: the base folder where all scenes are kept
        """
        if not os.path.exists(data_dir):
            raise RuntimeError(f"Cannot find data dir at {data_dir}")
        self.data_dir = data_dir
        self.verbose = verbose
        self.scenes = sorted(next(os.walk(data_dir))[1])
        self.splits_scenes = self.make_splits_scenes(
            seed=seed,
            frac_train=split_fracs["train"],
            frac_val=split_fracs["val"],
            frac_test=split_fracs["test"],
        )

    def get_scene_dataset_by_index(self, scene_idx):
        return CarlaSceneDataset(self.data_dir, self.scenes[scene_idx])

    def get_scene_dataset_by_name(self, scene_name):
        if not scene_name in self.scenes:
            raise IndexError(f"Cannot find scene {scene_name} in {self.scenes}")
        return CarlaSceneDataset(self.data_dir, scene_name)


class CarlaSceneDataset(BaseSceneDataset):
    name = "CARLA"
    CFG = {}
    CFG["num_lidar_features"] = 4
    nominal_whitelist_types = _nominal_whitelist_types
    nominal_ignore_types = _nominal_ignore_types

    def __init__(
        self,
        data_dir,
        scene,
        whitelist_types=_nominal_whitelist_types,
        ignore_types=_nominal_ignore_types,
    ):

        self.data_dir = data_dir
        self.scene = scene
        self.sequence_id = scene
        self.scene_path = os.path.join(data_dir, scene)

        # -- object and ego files
        self.obj_folder = os.path.join(self.scene_path, "objects")
        if not os.path.exists(os.path.join(self.obj_folder)):
            raise RuntimeError(
                "Could not find object folder {}".format(self.obj_folder)
            )
        self.obj_local_folder = os.path.join(self.scene_path, "objects_sensor")
        agent_files = {"timestamp": {}, "frame": {}}
        npc_files = {"timestamp": {}, "frame": {}}
        agent_frame_to_ts = {}
        npc_frame_to_ts = {}
        all_ts = set()
        all_frames = set()
        for filename in sorted(glob.glob(os.path.join(self.obj_folder, "*.txt"))):
            filename = filename.split("/")[-1]
            _, id_str, frame, ts = filename.split("-")
            ts = float(".".join(ts.split(".")[0:2]))
            frame = int(frame)
            if "actors" in id_str:
                agent_files["timestamp"][ts] = filename
                agent_files["frame"][frame] = filename
                agent_frame_to_ts[frame] = ts
            elif "npc" in id_str:
                # -- global frame
                npc_files["timestamp"][ts] = filename
                npc_files["frame"][frame] = filename
                npc_frame_to_ts[frame] = ts
            else:
                raise NotImplementedError(filename)
            all_frames.add(frame)
            all_ts.add(ts)
        self.agent_frame_to_ts = agent_frame_to_ts
        self.npc_frame_to_ts = npc_frame_to_ts
        self.agent_files = agent_files
        self.npc_files = npc_files

        # -- dynamically create sensor ID mappings
        sensor_IDs = {}
        sensor_folders = {}
        sensor_file_post = {}
        sensor_frame_to_ts = {}
        sensor_frames = {}
        file_endings = {}
        all_agent_IDs = set()
        sensor_data_folder = os.path.join(self.scene_path, "data")
        if os.path.exists(sensor_data_folder):
            self.sensor_data_folder = sensor_data_folder
            for sens in sorted(next(os.walk(self.sensor_data_folder))[1]):
                if not len(sens.split("-")) == 3:
                    raise ValueError(f"Cannot understand sensor data folder {sens}")
                name, sensor_ID, agent_ID = sens.split("-")
                agent_ID = int(agent_ID)
                all_agent_IDs.add(agent_ID)
                sensor_ID = f"{name}-{sensor_ID}"

                # make data structures
                if agent_ID not in sensor_IDs:
                    sensor_IDs[agent_ID] = []
                    sensor_folders[agent_ID] = {}
                    sensor_file_post[agent_ID] = {}
                    sensor_frame_to_ts[agent_ID] = {}
                    sensor_frames[agent_ID] = {}
                    file_endings[agent_ID] = {}

                # populate data structures
                sensor_IDs[agent_ID].append(sensor_ID)
                sensor_folders[agent_ID][sensor_ID] = os.path.join(
                    sensor_data_folder, sens
                )
                sensor_file_post[agent_ID][sensor_ID] = {"timestamp": {}, "frame": {}}
                sensor_frame_to_ts[agent_ID][sensor_ID] = {}
                sensor_frames[agent_ID][sensor_ID] = []

                # within each folder, parse the sensor timestamps and frames
                for i, filename in enumerate(
                    sorted(
                        glob.glob(
                            os.path.join(sensor_folders[agent_ID][sensor_ID], "data-*")
                        )
                    )
                ):
                    # parse timestamp and frames
                    filename = filename.split("/")[-1]
                    if i == 0:
                        file_endings[agent_ID][sensor_ID] = (
                            "." + filename.split(".")[-1]
                        )
                    _, ts, frame = filename.split("-")
                    ts = float(ts.split("_")[1])
                    frame = int(frame.split(".")[0].split("_")[1])
                    sensor_frame_to_ts[agent_ID][sensor_ID][frame] = float(ts)
                    sensor_frames[agent_ID][sensor_ID].append(frame)

                    # save the filename
                    fname_post = filename.replace("data-", "")[:-4]
                    if fname_post.endswith("."):
                        fname_post = fname_post[:-1]
                    sensor_file_post[agent_ID][sensor_ID]["timestamp"][ts] = fname_post
                    sensor_file_post[agent_ID][sensor_ID]["frame"][frame] = fname_post

                sensor_frames[agent_ID][sensor_ID] = sorted(
                    sensor_frames[agent_ID][sensor_ID]
                )
        else:
            raise FileNotFoundError(
                "Cannot find data folder {}".format(sensor_data_folder)
            )
        self.frames = np.asarray(sorted(all_frames))
        self.timestamps = np.asarray(sorted(all_ts))
        self.agent_IDs = sorted(all_agent_IDs)
        self.sensor_IDs = sensor_IDs
        self.sensor_folders = sensor_folders
        self.sensor_file_post = sensor_file_post
        self.sensor_frame_to_ts = sensor_frame_to_ts
        self.sensor_frames = sensor_frames
        self.file_endings = file_endings
        super().__init__(whitelist_types, ignore_types)

    def __str__(self):
        return f"CARLA Object Datset of folder: {self.scene_path}"

    def get_object_file(
        self, frame, timestamp, is_agent, is_global, agent=None, sensor=None
    ):
        check_xor_for_none(frame, timestamp)
        if frame is not None:
            if is_agent:
                file_post = self.agent_files["frame"][frame]
            else:
                file_post = self.npc_files["frame"][frame]
                if not is_global:
                    file_post = file_post.replace("npcs", "objects")
        else:
            raise
        if is_global:
            filepath = os.path.join(self.obj_folder, file_post)
        else:
            if sensor is None:
                filepath = os.path.join(
                    self.obj_local_folder, f"agent-{agent}", file_post
                )
            else:
                filepath = os.path.join(
                    self.obj_local_folder, f"{sensor}-{agent}", file_post
                )
        return filepath

    def get_sensor_file(self, frame, timestamp, sensor, agent, file_type):
        check_xor_for_none(frame, timestamp)
        if agent is None:
            raise ValueError(agent)
        elif sensor is None:
            raise ValueError(sensor)
        if frame is not None:
            file_post = self.sensor_file_post[agent][sensor]["frame"][frame]
        else:
            # TODO: ALLOW FOR INTERPOLATION OR NEAREST????
            file_post = self.sensor_file_post[agent][sensor]["timestamp"][timestamp]
        filepath = os.path.join(
            self.sensor_folders[agent][sensor], file_type + "-" + file_post
        )
        return filepath

    def get_sensor_name(self, sensor, agent):
        return sensor

    def get_agents(self, frame: int) -> "DataContainer":
        return self._load_agents(frame)

    def get_agent(self, frame: int, agent: int):
        agents = self.get_agents(frame)
        return [ag for ag in agents if ag.ID == agent][0]

    def _load_agent_set(self, frame: int) -> set:
        # TODO: this is slow...improve
        return {ag.ID for ag in self.get_agents(frame)}

    def get_sensor_ID(self, sensor: str, agent: int):
        return f"{sensor}-{agent}"

    def get_ego_reference(self, *args, **kwargs):
        return None

    def _load_agents(self, frame):
        timestamp = None
        if frame is None:
            frame = self.get_frames(
                sensor="camera-0",
                agent=0,
            )[0]
        filepath = self.get_object_file(frame, timestamp, is_agent=True, is_global=True)
        return read_agents_from_file(filepath)

    def _load_sensor_data_filepath(self, frame, sensor, agent):
        return (
            self.get_sensor_file(frame, None, sensor, agent, "data")
            + self.file_endings[agent][sensor]
        )

    def _load_frames(self, sensor: str, agent: int):
        return self.sensor_frames[agent][sensor]

    def _load_timestamp(self, frame, sensor=None, agent=None, utime=False):
        if sensor is None:
            return self.timestamps[self.frames == frame][0]
        else:
            return self.sensor_frame_to_ts[agent][sensor][frame]

    def _load_calibration(self, frame, sensor, agent, *args, **kwargs):
        timestamp = None
        filepath = (
            self.get_sensor_file(frame, timestamp, sensor, agent, "calib") + ".txt"
        )
        with open(filepath, "r") as f:
            calib = json.load(f, cls=calibration.CalibrationDecoder)
        return calib

    def _load_im_general(self, frame, sensor, agent):
        timestamp = None
        filepath = (
            self.get_sensor_file(frame, timestamp, sensor, agent, "data")
            + self.file_endings[agent][sensor]
        )
        assert os.path.exists(filepath), filepath
        try:
            return imread(filepath)
        except TypeError as e:
            print(filepath)
            raise e

    def _load_image(self, frame, sensor, agent):
        return self._load_im_general(frame, sensor, agent)

    def _load_semseg_image(self, frame, sensor, agent):
        return self._load_im_general(frame, sensor, agent)

    def _load_depth_image(self, frame, sensor, agent):
        return self._load_im_general(frame, sensor, agent)

    def _load_lidar(self, frame, sensor, agent, filter_front, with_panoptic=False):
        timestamp = None
        filepath = (
            self.get_sensor_file(frame, timestamp, sensor, agent, "data")
            + self.file_endings[agent][sensor]
        )
        assert os.path.exists(filepath), filepath
        pcd = read_pc_from_file(
            filepath,
            n_features=self.CFG["num_lidar_features"],
            filter_front=filter_front,
        )
        return pcd

    def _load_radar(self, frame, sensor, agent):
        timestamp = None
        filepath = (
            self.get_sensor_file(frame, timestamp, sensor, agent, "data")
            + self.file_endings[agent][sensor]
        )
        assert os.path.exists(filepath), filepath
        rad = np.fromfile(filepath, dtype=np.float32).reshape((-1, 4))
        return rad

    def _load_objects(
        self,
        frame,
        sensor,
        agent,
        whitelist_types=["car", "truck", "bicycle", "motorcycle"],
        ignore_types=[],
    ):
        timestamp = None
        filepath = self.get_object_file(
            frame,
            timestamp,
            sensor=sensor,
            agent=agent,
            is_agent=False,
            is_global=False,
        )
        objs = read_objects_from_file(filepath)
        return DataContainer(
            frame=frame,
            timestamp=self.get_timestamp(frame=frame, sensor=sensor, agent=agent),
            data=[
                obj
                for obj in objs
                if ((obj.obj_type in whitelist_types) or (whitelist_types == "all"))
                and (obj.obj_type not in ignore_types)
            ],
            source_identifier=f"{sensor}-{agent}",
        )

    def _load_objects_global(
        self,
        frame,
        whitelist_types="all",
        ignore_types=[],
        include_agents=True,
        ignore_static_agents=True,
        max_dist: Union[Tuple[ReferenceFrame, float], None] = None,
    ):
        timestamp = None
        filepath = self.get_object_file(
            frame, timestamp, is_agent=False, is_global=True
        )
        objs = read_objects_from_file(filepath)
        objs = list(objs) if objs.size > 0 else []

        # load the ego objects as well
        if include_agents:
            agents_as_objects = [
                agent.change_reference(GlobalOrigin3D, inplace=False)
                for agent in self.get_agents(frame)
            ]
            # HACK: alter the ID of the agents to mitigate conflict
            for agent in agents_as_objects:
                agent.ID += 99999
            if ignore_static_agents:
                agents_as_objects = [
                    agent
                    for agent in agents_as_objects
                    if "static" not in agent.obj_type
                ]
            objs.extend(agents_as_objects)

        # filter objects if they fit max distance
        if max_dist:
            objs = [
                obj
                for obj in objs
                if obj.position.change_reference(max_dist[0], inplace=False).norm()
                < max_dist[1]
            ]

        return np.array(
            [
                obj
                for obj in objs
                if ((obj.obj_type in whitelist_types) or (whitelist_types == "all"))
                and (obj.obj_type not in ignore_types)
            ]
        )

    def _number_objects_from_file(self, frame, **kwargs):
        timestamp = None
        filepath = self.get_object_file(
            frame, timestamp, is_agent=False, is_global=True
        )
        with open(filepath, "r") as f:
            lines = f.readlines()
        return len(lines)

    def _save_objects(self, frame, objects, folder, file):
        with open(os.path.join(folder, file.format("txt")), "w") as f:
            f.write(objects.encode())


def read_agents_from_file(filepath):
    DC = DataContainerDecoder
    DC.data_decoder = ObjectStateDecoder
    with open(filepath, "r") as f:
        agents = json.load(f, cls=DC)
    return agents


def read_objects_from_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1
    DC = DataContainerDecoder
    DC.data_decoder = ObjectStateDecoder
    objs = json.loads(lines[0], cls=DC)
    return np.asarray(objs.data)


def read_pc_from_file(filepath, n_features: int, filter_front: bool):
    if filepath.endswith(".ply"):
        pcd = o3d.io.read_point_cloud(filepath)
        pcd = np.asarray(pcd.points)
    else:
        pcd = np.fromfile(filepath, dtype=np.float32).reshape((-1, n_features))
    if filter_front:
        return pcd[pcd[:, 0] > 0, :]  # assumes z is forward....
    else:
        return pcd


def read_calibration_from_file(filepath):
    with open(filepath, "r") as f:
        calib = json.load(f, cls=calibration.CalibrationDecoder)
    return calib
