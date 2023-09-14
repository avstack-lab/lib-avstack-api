# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-09-12
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-12
# @Description:
"""

"""

try:
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent
except ModuleNotFoundError as e:
    print("Cannot run the leaderboard evaluation")

try:
    from srunner.autoagents.autonomous_agent import Track
except ModuleNotFoundError as e:
    print("Scenario runner could not be found")


def get_entry_point():
    return "CarlaLeaderboardAgent"


class CarlaLeaderboardAgent(AutonomousAgent):
    """A wrapper to be able to run our agent on the carla challenge"""

    def sensors(self):
        """Defines the sensors used for our AV

        NOTE: coordinates must be in the unreal frame:
            x - forward
            y - right
            z - up

        Maximum number of sensors are as follows:
            sensor.camera.rgb: 4
            sensor.lidar.ray_cast: 1
            sensor.other.radar: 2
            sensor.other.gnss: 1
            sensor.other.imu: 1
            sensor.opendrive_map: 1
            sensor.speedometer: 1

        If a sensor is located more than 3 meters away from its
        parent in any axis (e.g. [3.1,0.0,0.0]), the setup will fail
        """
        sensors = [
            {
                "type": "sensor.camera.rgb",
                "id": "Center",
                "x": 0.7,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 300,
                "height": 200,
                "fov": 100,
            },
            {
                "type": "sensor.lidar.ray_cast",
                "id": "LIDAR",
                "x": 0.7,
                "y": -0.4,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -45.0,
            },
            {
                "type": "sensor.other.radar",
                "id": "RADAR",
                "x": 0.7,
                "y": -0.4,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -45.0,
                "fov": 30,
            },
            {"type": "sensor.other.gnss", "id": "GPS", "x": 0.7, "y": -0.4, "z": 1.60},
            {
                "type": "sensor.other.imu",
                "id": "IMU",
                "x": 0.7,
                "y": -0.4,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -45.0,
            },
            # {'type': 'sensor.opendrive_map', 'id': 'OpenDRIVE', 'reading_frequency': 1},
            {"type": "sensor.speedometer", "id": "Speed"},
        ]
        return sensors

    def setup(self, path_to_conf_file, track=Track.SENSORS):
        """At a minimum, this method sets the Leaderboard modality.
        In this case, SENSORS"""
        self.track = track

    def run_step(self, input_data, timestamp):
        """
        input_data: A dictionary containing sensor data
        for the requested sensors. The data has been preprocessed
        at sensor_interface.py, and will be given as numpy arrays.
        This dictionary is indexed by the ids defined in the sensor method.

        timestamp: A timestamp of the current simulation instant.

        Remember that you also have access to the route that the
        ego agent should travel to achieve its destination.
        Use the self._global_plan member to access the geolocation
        route and self._global_plan_world_coord for its world location counterpart.
        """
        control = self._do_something_smart(input_data, timestamp)
        return control

    def destroy(self):
        """At the end of each route, the destroy method will be called,
        which can be overriden by your agent, in cases where you need a
        cleanup. As an example, you can make use of this function to
        erase any unwanted memory of a network"""
