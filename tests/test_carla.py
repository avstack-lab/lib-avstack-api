# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-09-28
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-28
# @Description:
"""

"""

import os, logging
from avapi.carla import CarlaScenesManager


carla_data_dir = './data/CARLA'
if os.path.exists(carla_data_dir):
    if len(os.listdir('/your/path')) > 0:
        CSM = CarlaScenesManager(carla_data_dir)
    else:
        CSM = None
        msg = "Cannot run test - CARLA data not downloaded"
else:
    CSM = None
    msg = "Cannot run test - CARLA data not downloaded"


def test_csm_list_scenes():
    if CSM is not None:
        CSM.list_scenes()
    else:
        logging.warning(msg)
        print(msg)