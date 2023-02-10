# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2021-10-25
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-10-22
# @Description:
"""
Code usefule for getting carla situations started
"""
import logging
import os
import random
import time

import carla
import numpy as np

from avapi.carla import config
from avapi.carla.simulator import sensors, utils
from avapi.carla.simulator.ego import CarlaEgoActor
from avapi.carla.simulator.manager import CarlaManager, InfrastructureManager


# -------------------------------------------------------------
# Bootstrap commands -- these will usually take in a client and params file
# -------------------------------------------------------------


def bootstrap_client(cfg=None, config_file="./default_client.yml"):
    if cfg is None:
        cfg = config.read_config(config_file)

    try:
        client = carla.Client(cfg["connect_ip"], cfg["connect_port"])
        client.set_timeout(2.0)
        traffic_manager = client.get_trafficmanager(cfg["traffic_manager_port"])
        traffic_manager.set_synchronous_mode(cfg["synchronous"])
    except Exception as e:
        raise e  # to handle errors in the future

    world = client.get_world()
    orig_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = cfg["synchronous"]
    settings.fixed_delta_seconds = 1.0 / cfg["rate"]
    world.apply_settings(settings)
    return client, world, traffic_manager, orig_settings


def apply_world_settings(world, cfg):
    # -- get walker info
    blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")
    n_walkers = cfg["n_random_walkers"]
    spawn_points = []
    for i in range(n_walkers):
        spawn_points.append(
            carla.Transform(
                world.get_random_location_from_navigation(), carla.Rotation()
            )
        )
    npcs_walk = _spawn_agents(world, blueprints, spawn_points, n_walkers)

    # -- get vehicle info
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    n_vehicles = cfg["n_random_vehicles"]
    npcs_veh = _spawn_agents(world, blueprints, spawn_points, n_vehicles)

    time.sleep(2)
    return npcs_walk + npcs_veh


def _spawn_agents(world, blueprints, spawn_points, n_agents):
    # Check the number of agents
    if n_agents < len(spawn_points):
        random.shuffle(spawn_points)
    elif n_agents > len(spawn_points):
        msg = "requested %d agents, but could only find %d spawn points"
        logging.warning(msg, n_agents, len(spawn_points))
        n_agents = len(spawn_points)
    # -- spawn all agents
    walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    npcs = []
    i_succ = 0
    try:
        print("Spawning %d npcs randomly" % n_agents)
        for i in range(n_agents):
            bp = np.random.choice(blueprints)
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            if bp.has_attribute("driver_id"):
                driver_id = random.choice(
                    bp.get_attribute("driver_id").recommended_values
                )
                bp.set_attribute("driver_id", driver_id)
            npc = world.try_spawn_actor(bp, spawn_points[i])
            if npc is not None:
                if "walker" in npc.type_id:
                    ai_controller = world.try_spawn_actor(
                        walker_controller_bp, carla.Transform(), npc
                    )
                    ai_controller.start()
                    ai_controller.go_to_location(
                        world.get_random_location_from_navigation()
                    )
                    ai_controller.set_max_speed(
                        1 + random.random()
                    )  # Between 1 and 2 m/s (default is 1.4 m/s).
                else:
                    npc.set_autopilot(True)
                npcs.append(npc)
                i_succ += 1
    except Exception as e:
        for npc in npcs:
            npc.destroy()
        raise e
    print("Successfully spawned %i npcs" % i_succ)
    return npcs


def bootstrap_display(world, ego, cfg=None, config_file="./default_display.yml"):
    if cfg is None:
        cfg = config.read_config(config_file)

    if cfg["enabled"]:
        print("Enabled display")
        import pygame

        from . import display

        # -- display
        pg_display = pygame.display.set_mode(
            (cfg["disp_x"], cfg["disp_y"]), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        # -- display manager
        hud = display.HUD(cfg["disp_x"], cfg["disp_y"])
        world.on_tick(hud.on_world_tick)
        display_manager = display.CameraDisplayManager(
            ego, pg_display, hud, gamma_correction=2.2
        )
        try:
            display_manager.set_sensor(index=0, notify=False)
            # -- keyboard controls
            keyboard_control = display.KeyboardControl(display_manager)
        except Exception as e:
            display_manager.destroy()
            raise e
    else:
        print("Display is not enabled for this run")
        display_manager = None
        keyboard_control = None
    return display_manager, keyboard_control


sensor_options = {
    "camera": sensors.RgbCameraSensor,
    "gnss": sensors.GnssSensor,
    "gps": sensors.GnssSensor,
    "depthcam": sensors.DepthCameraSensor,
    "imu": sensors.ImuSensor,
    "lidar": sensors.LidarSensor,
    "radar": sensors.RadarSensor,
}


def bootstrap_ego(
    world, ego_stack, cfg=None, config_file="./default_ego.yml", save_folder=""
):
    if cfg is None:
        cfg = config.read_config(config_file)

    # --- make ego
    ego = CarlaEgoActor(world, ego_stack, cfg)

    # --- make sensors attached to ego
    try:
        for sens in sensor_options.values():
            sens.reset_next_id()
        for i, cfg_sensor in enumerate(cfg["sensors"]):
            bootstrap_ego_sensor(ego, i, cfg_sensor, save_folder)
    except Exception as e:
        ego.destroy()
        raise e

    # --- make other sensors
    pass

    return ego


def bootstrap_infrastructure(
    world, cfg, config_file="./default_infrastructure.yml", save_folder=""
):
    if cfg is None:
        cfg = config.read_config(config_file)

    # import ipdb; ipdb.set_trace()
    # -- infrastructure class to act like "parent" actor
    infra = InfrastructureManager(world)

    # -- make sensors
    infra_sensors = {k: [] for k in cfg}
    n_infra_spawn = 0
    try:
        for k in cfg:
            for idx in range(cfg[k]["n_spawn"]):
                bootstrap_infra_sensor(infra, idx, cfg[k], save_folder=save_folder)
                n_infra_spawn += 1
    except Exception as e:
        infra.destroy()
        raise e
    print("Spawned %i infrastructure elements" % n_infra_spawn)
    return infra


def bootstrap_infra_sensor(infra, idx, cfg, save_folder):
    # -- find spawn point
    spawn_points = infra.map.get_spawn_points()
    if cfg["idx_spawn"] == "random":
        spawn_point = random.choice(spawn_points)
    elif cfg["idx_spawn"] == "in_order":
        spawn_point = spawn_points[cfg["idx_spawn_list"][idx]]
    else:
        spawn_point = (
            spawn_points[cfg["idx_spawn"]]
            if cfg["idx_spawn"]
            else random.choice(spawn_points)
        )
    x_spawn = utils.carla_location_to_numpy_vector(
        spawn_point.location
    ) + utils.carla_location_to_numpy_vector(
        carla.Location(
            x=cfg["transform"]["location"]["x"],
            y=cfg["transform"]["location"]["y"],
            z=cfg["transform"]["location"]["z"],
        )
    )
    x_spawn[2] -= spawn_point.location.z  # only allow for the manual z component
    q_spawn = utils.carla_rotation_to_quaternion(
        carla.Rotation(
            pitch=cfg["transform"]["rotation"]["pitch"],
            yaw=cfg["transform"]["rotation"]["yaw"],
            roll=cfg["transform"]["rotation"]["roll"],
        )
    ) * utils.carla_rotation_to_quaternion(spawn_point.rotation)

    tform_spawn = carla.Transform(
        utils.numpy_vector_to_carla_location(x_spawn),
        utils.quaternion_to_carla_rotation(q_spawn),
    )

    # -- spawn sensor
    save_folder = os.path.join(save_folder, "sensor_data")
    sens = sensor_options[cfg["sensor_name"]]
    source_name = cfg["name_prefix"] + f"_{idx+1:03d}"
    pos_covar = cfg["position_uncertainty"]
    sens = sens(
        source_name,
        infra,
        tform_spawn,
        cfg["attributes"],
        cfg["mode"],
        cfg["noise"],
        save=cfg["save"],
        save_folder=save_folder,
    )

    # -- add to infra
    infra.add_sensor(
        source_name, sens, comm_range=cfg["comm_range"], pos_covar=pos_covar
    )


def bootstrap_ego_sensor(ego, ID, cfg, save_folder):
    # --- make sensor
    assert isinstance(cfg, dict)
    if len(cfg) != 1:
        import ipdb

        ipdb.set_trace()
        raise RuntimeError
    k1 = list(cfg.keys())[0]
    source_name = cfg[k1]["name"]
    save_folder = os.path.join(save_folder, "sensor_data")
    tform = carla.Transform(
        carla.Location(
            cfg[k1]["transform"]["location"]["x"],
            cfg[k1]["transform"]["location"]["y"],
            cfg[k1]["transform"]["location"]["z"],
        ),
        carla.Rotation(
            cfg[k1]["transform"]["rotation"]["pitch"],
            cfg[k1]["transform"]["rotation"]["yaw"],
            cfg[k1]["transform"]["rotation"]["roll"],
        ),
    )
    for k, sens in sensor_options.items():
        if k in k1:
            break
    else:
        raise NotImplementedError(k1)
    ego.add_sensor(
        k1,
        sens(
            source_name,
            ego,
            tform,
            cfg[k1]["attributes"],
            cfg[k1]["mode"],
            cfg[k1]["noise"],
            save=cfg[k1]["save"],
            save_folder=save_folder,
        ),
    )


def bootstrap_npcs(world, cfg, verbose=False):
    npcs = []
    npc_cfgs = []

    # -- add npcs that for some reason carla decided to auto-create...
    # actors_pre = world.get_actors()
    # whitelists = ['vehicle', 'walker']
    # for actor in actors_pre:
    #     if sum([w in actor.type_id for w in whitelists]) > 0:
    #         npcs.append(actor)
    #         npc_cfgs.append(actor.attributes)

    # -- add npcs from the world
    for item in cfg:
        if not "npc" in item:
            continue
        # -- spawn
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = (
            spawn_points[cfg[item]["idx_spawn"]]
            if cfg[item]["idx_spawn"]
            else random.choice(spawn_points)
        )
        bp = world.get_blueprint_library().filter(cfg[item]["type"])
        if cfg[item]["idx_vehicle"] is None:
            bp_choice = random.choice(bp)
        else:
            bp_choice = bp[cfg[item]["idx_vehicle"]]
        d_spawn = cfg[item]["delta_spawn"]
        spawn_point.location.x += d_spawn["x"]
        spawn_point.location.y += d_spawn["y"]
        spawn_point.location.z += d_spawn["z"]
        try:
            npcs.append(world.spawn_actor(bp_choice, spawn_point))
        except RuntimeError as e:
            print("Removing npcs")
            for npc in npcs:
                npc.destroy()
            raise e
        npc_cfgs.append(cfg[item])
        # -- destination
        if cfg[item]["autopilot"]:
            npcs[-1].set_autopilot()
        if verbose:
            print("Spawned NPC")
    return npcs, npc_cfgs


# -------------------------------------------------------------
# Bootstrap for test cases
# -------------------------------------------------------------


def bootstrap_standard(world, traffic_manager, ego_stack, cfg, save_folder):
    if cfg is None:
        cfg = config.read_config(config_file)

    # -- unload parked cars!
    try:
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    except AttributeError as e:
        pass  # some version do not support this

    ego = bootstrap_ego(world, ego_stack, cfg["ego"], save_folder=save_folder)
    npcs_random = apply_world_settings(world, cfg["world"])
    try:
        npcs_set, npc_cfgs = bootstrap_npcs(world, cfg)
    except Exception as e:
        ego.destroy()
        raise e
    try:
        infra = bootstrap_infrastructure(
            world, cfg["infrastructure"], save_folder=save_folder
        )
    except Exception as e:
        ego.destroy()
        for npc in npcs_set:
            npc.destroy()
        raise e

    manager = CarlaManager(
        world,
        traffic_manager,
        record_truth=cfg["recorder"]["record_truth"],
        record_folder=save_folder,
    )
    manager.ego = ego
    manager.npcs = npcs_random + npcs_set
    manager.infrastructure = infra
    try:
        manager.schedule_npc_events(npc_cfgs)
    except Exception as e:
        manager.destroy()
        raise e
    return manager


# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
