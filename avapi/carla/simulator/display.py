# -*- coding: utf-8 -*-
# @Author: spencer
# @Date:   2020-12-26
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-28
# @Description: Utilities for carla applications

"""
Welcome to the Heads Up Display HELP for lib-carcar

Use Keys to toggle modes:
    
    V          : disabled...toggle view of the HUD camera, if HUD selected
    S          : disabled...toggle the sensor used for viewing
    R          : change the representation of objects on the screen
    SHIFT + M  : enter manual control mode
    SHIFT + A  : enter autopilot control mode


If the ego is in MANUAL mode, use ARROWS for control

    UP     : throttle
    DOWN   : brake
    LEFT   : steer left
    RIGHT  : steer right
    Q      : hold to enter reverse

"""

import datetime
import math
import os
import sys
import weakref

import carla
import cv2
import numpy as np
import pygame
from avstack import calibration
from avstack import transformations as tforms
from avstack.geometry import Origin, Rotation, Translation, bbox
from avstack.utils import maskfilters
from carla import ColorConverter as cc
from pygame.locals import (
    K_0,
    K_9,
    K_BACKQUOTE,
    K_BACKSPACE,
    K_COMMA,
    K_DOWN,
    K_EQUALS,
    K_ESCAPE,
    K_F1,
    K_LEFT,
    K_MINUS,
    K_PERIOD,
    K_RIGHT,
    K_SLASH,
    K_SPACE,
    K_TAB,
    K_UP,
    KMOD_CTRL,
    KMOD_SHIFT,
    K_a,
    K_c,
    K_d,
    K_g,
    K_h,
    K_i,
    K_l,
    K_m,
    K_n,
    K_p,
    K_q,
    K_r,
    K_s,
    K_v,
    K_w,
    K_x,
    K_z,
)

from avapi.carla import utils
from avapi.visualize import draw_box2d, draw_projected_box3d


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    name = lambda x: " ".join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


# ==============================================================================
# -- Camera Display Manager ----------------------------------------------------
# ==============================================================================


class CameraDisplayManager(object):
    def __init__(self, parent, pg_display, hud, gamma_correction):
        self._parent = parent
        self.pg_display = pg_display
        self.hud = hud
        self.gamma_correction = gamma_correction
        self.recording = False
        self.surface = None
        self.sensor = None
        self.objects = {}
        self.object_representations = ["object_3d", "object_2d", "track_3d", "off"]
        self.representation_colors = {
            "object_3d": (0, 0, 255),
            "object_2d": (0, 255, 0),
            "track_3d": (255, 255, 255),
        }
        self.idx_representation = 2 - 1
        self.toggle_obj_representation()
        self.R_UE2C = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]).T
        bound_y = 0.5 + self._parent.actor.bounding_box.extent.y

        # -- set the allowable cameras
        self.sensor_view = "hud"  # TODO: allow for changing this index
        self._camera_transforms = {}
        self._camera_Ps = {}
        self._camera_bbox = {}

        self.add_hud_camera()
        self.add_ego_cameras()

        # -- save the sensor type to be used
        self.transform_index = 0
        self.index = 0
        self.sensors = [["sensor.camera.rgb", cc.Raw, "Camera RGB", {}]]
        self.set_cameras()

    def set_cameras(self):
        bp_library = self._parent.actor.get_world().get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                bp.set_attribute("image_size_x", str(self.hud.dim[0]))
                bp.set_attribute("image_size_y", str(self.hud.dim[1]))
                if bp.has_attribute("gamma"):
                    bp.set_attribute("gamma", str(self.gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith("sensor.lidar"):
                self.lidar_range = 50
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == "range":
                        self.lidar_range = float(attr_value)
            item.append(bp)
        self.set_sensor(self.index, self.sensor_view, force_respawn=True)

    def add_camera(self, name, P, tform, attachment):
        if name not in self._camera_transforms:
            self._camera_transforms[name] = []
            self._camera_Ps[name] = []
            self._camera_bbox[name] = []
        self._camera_transforms[name].append((tform, attachment))
        self._camera_Ps[name].append(P)
        q = utils.carla_rotation_to_quaternion(tform.rotation)
        x = utils.carla_location_to_numpy_vector(tform.location)
        origin = Origin(x, q)
        img_shape = (2 * P[1, 2], 2 * P[0, 2])
        cam_calib = calibration.CameraCalibration(origin, P, img_shape)
        self._camera_bbox[name].append(
            bbox.Box2D([0, 0, img_shape[0], img_shape[1]], cam_calib)
        )

    def add_hud_camera(self):
        """Add the HUD camera to the allowable views"""
        # Make the camera transformation matrix relative to center of car
        w = self.hud.dim[0]
        h = self.hud.dim[1]
        fov = 90.0  # horizontal FOV
        f = w / (2 * np.tan(fov * math.pi / 360.0))
        camera_P = np.array([[f, 0, w / 2.0, 0], [0, f, h / 2.0, 0], [0, 0, 1, 0]])

        self.add_camera(
            "hud",
            camera_P,
            carla.Transform(carla.Location(x=-8.5, z=3.5)),
            carla.AttachmentType.Rigid,
        )
        self.add_camera(
            "hud",
            camera_P,
            carla.Transform(carla.Location(z=15), carla.Rotation(pitch=-90)),
            carla.AttachmentType.Rigid,
        )

    def add_ego_cameras(self):
        """Add a replica of the ego camera to the allowable views"""
        for name, sensor in self._parent.sensors.items():
            if "camera" in name:
                P = sensor.P
                T = sensor.tform_to_parent
                self.add_camera(name, P, T, carla.AttachmentType.Rigid)

    def tick(self, world, ego, clock):
        self.hud.tick(world, ego, clock)

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.destroy()
            except RuntimeError as e:
                # usually because it was already destroyed somehow
                pass
            finally:
                self.sensor = None

    def restart(self, ego):
        self.destroy()
        self._parent = ego
        self.set_cameras()

    def render(self):
        if self.surface is not None:
            self.pg_display.blit(self.surface, (0, 0))
        self.hud.render(self.pg_display)
        pygame.display.flip()

    def toggle_view(self):
        self.transform_index = (self.transform_index + 1) % len(
            self._camera_transforms[self.sensor_view]
        )
        self.set_sensor(
            self.index, name=self.sensor_view, notify=False, force_respawn=True
        )

    def set_sensor(self, index, name="hud", notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = (
            True
            if self.index is None
            else (
                force_respawn or (self.sensors[index][2] != self.sensors[self.index][2])
            )
        )
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.actor.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[name][self.transform_index][0],
                attach_to=self._parent.actor,
                attachment_type=self._camera_transforms[name][self.transform_index][1],
            )
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraDisplayManager._parse_image(weak_self, image)
            )
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))

    def set_objects(self, objects, tracks):
        self._set_objects("object_2d", objects.get("object_2d", []))
        self._set_objects("object_3d", objects.get("object_3d", []))
        self._set_objects("track_3d", tracks)

    def _set_objects(self, key, objs):
        self.objects[key] = objs

    def toggle_obj_representation(self):
        self.idx_representation = (self.idx_representation + 1) % len(
            self.object_representations
        )
        print(
            f"Representation set as: {self.object_representations[self.idx_representation]}"
        )

    def add_objects_to_image(self, img):
        # TODO: this is not quite right but almost works...
        img1 = img.astype(np.uint8).copy()
        representation = self.object_representations[self.idx_representation]
        if (representation in self.objects) and (representation != "off"):
            for obj in self.objects[representation]:
                pass
        return img1

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith("sensor.camera.dvs"):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(
                image.raw_data,
                dtype=np.dtype(
                    [
                        ("x", np.uint16),
                        ("y", np.uint16),
                        ("t", np.int64),
                        ("pol", np.bool),
                    ]
                ),
            )
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[
                dvs_events[:]["y"], dvs_events[:]["x"], dvs_events[:]["pol"] * 2
            ] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            # --- add object boxes
            array = self.add_objects_to_image(array)
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)


class KeyboardControl(object):
    """class that handles keyboard input"""

    def __init__(self, display_manager):
        display_manager.hud.notification("Press 'H' for help", seconds=3.0)
        self._display_manager = display_manager

    def parse_events(self, world):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_r:
                    self._display_manager.toggle_obj_representation()
                elif event.key == K_v:
                    self._display_manager.toggle_view()
                elif event.key == K_s:
                    self._display_manager.toggle_sensor()
                elif event.key == K_h or (
                    event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT
                ):
                    self._display_manager.hud.help.toggle()
                elif event.key == K_m and pygame.key.get_mods() & KMOD_SHIFT:
                    self._display_manager._parent.set_control_mode("manual")
                elif event.key == K_a and pygame.key.get_mods() & KMOD_SHIFT:
                    self._display_manager._parent.set_control_mode("autopilot")

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- Heads Up Display ----------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width=800, height=600):
        pygame.init()
        pygame.font.init()

        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = "courier" if os.name == "nt" else "mono"
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == "nt" else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self.simulation_time_elapsed = 0
        self.world_map = None
        self.disp_name = None
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        if self.simulation_time == 0:
            self.simulation_time = timestamp.elapsed_seconds
        self.simulation_time_elapsed += timestamp.elapsed_seconds - self.simulation_time
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, ego, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        if self.world_map is None:
            self.world_map = world.get_map().name
        if self.disp_name is None:
            self.disp_name = get_actor_display_name(ego.actor, truncate=20)
        t_truth = ego.actor.get_transform()
        v_truth = ego.actor.get_velocity()
        c_truth = ego.actor.get_control()

        # Set IMU fields
        try:
            accel = ego.imu_sensor.accelerometer
            gyro = ego.imu_sensor.gyroscope
            compass = ego.imu_sensor.compass
        except AttributeError:
            accel = (-1, -1, -1)
            gyro = (-1, -1, -1)
            compass = 0
        heading = "N" if compass > 270.5 or compass < 89.5 else ""
        heading += "S" if 90.5 < compass < 269.5 else ""
        heading += "E" if 0.5 < compass < 179.5 else ""
        heading += "W" if 180.5 < compass < 359.5 else ""

        # Set GNSS fields
        try:
            lat, lon = (ego.gnss_sensor.lat, ego.gnss_sensor.lon)
        except AttributeError:
            lat, lon = (0, 0)

        # TODO: FIX THIS
        # colhist = ego.collision_sensor.get_collision_history()
        # collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        collision = [0]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.get_actors().filter("vehicle.*")

        # Get the integrity monitoring value
        try:
            integrity_stat = ego.navigator.system_integrity
        except AttributeError:
            integrity_stat = 0

        # Creat the test for the display
        self._info_text = [
            "Server:  % 16.0f FPS" % self.server_fps,
            "Client:  % 16.0f FPS" % clock.get_fps(),
            "",
            "Vehicle: % 20s" % self.disp_name,
            "Map:     % 20s" % self.world_map,
            "Simulation time: % 12s"
            % datetime.timedelta(seconds=int(self.simulation_time)),
            "",
            "Speed:   % 15.0f m/s"
            % (math.sqrt(v_truth.x**2 + v_truth.y**2 + v_truth.z**2)),
            "Location (avstack):% 20s"
            % ("(% 5.1f, % 5.1f)" % (t_truth.location.x, -t_truth.location.y)),
            "Height:  % 18.0f m" % t_truth.location.z,
            "",
            "System Integrity:",
            ("", integrity_stat, 0.0, 1.0),
            "",
        ]

        if isinstance(c_truth, carla.VehicleControl):
            self._info_text += [
                ("Throttle:", c_truth.throttle, 0.0, 1.0),
                ("Steer:", c_truth.steer, -1.0, 1.0),
                ("Brake:", c_truth.brake, 0.0, 1.0),
                ("Reverse:", c_truth.reverse),
                ("Hand brake:", c_truth.hand_brake),
                ("Manual:", c_truth.manual_gear_shift),
                "Gear:        %s" % {-1: "R", 0: "N"}.get(c_truth.gear, c_truth.gear),
            ]
        elif isinstance(c_truth, carla.WalkerControl):
            self._info_text += [
                ("Speed:", c_truth.speed, 0.0, 5.556),
                ("Jump:", c_truth.jump),
            ]
        self._info_text += [
            "",
            "Collision:",
            collision,
            "",
            "Number of vehicles: % 8d" % len(vehicles),
        ]
        if len(vehicles) > 1:
            self._info_text += ["Nearby vehicles:"]
            distance = lambda l: math.sqrt(
                (l.x - t_truth.location.x) ** 2
                + (l.y - t_truth.location.y) ** 2
                + (l.z - t_truth.location.z) ** 2
            )
            vehicles = [
                (distance(x.get_location()), x)
                for x in vehicles
                if x.id != ego.actor.id
            ]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append("% 4dm %s" % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text("Error: %s" % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [
                            (x + 8, v_offset + 8 + (1.0 - y) * 30)
                            for x, y in enumerate(item)
                        ]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect, 0 if item[1] else 1
                        )
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6)
                        )
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8),
                                (6, 6),
                            )
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6)
                            )
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split("\n")
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)
