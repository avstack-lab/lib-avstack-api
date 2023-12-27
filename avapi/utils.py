# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-04-03
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-11
# @Description:
"""

"""

import os


def get_timestamps(src_folder):
    ts = []
    with open(os.path.join(src_folder, "timestamps.txt")) as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            ts.append(line)
    return ts


def check_xor_for_none(a, b):
    assert check_xor(a is None, b is None), "Can only pass in one of these inputs"


def check_xor(a, b):
    return (a or b) and (not a or not b)


def remove_glob(glob_files):
    rem = False
    for f in glob_files:
        os.remove(f)
        rem = True
    if rem:
        print("Removed files from: {}".format(os.path.dirname(f)), flush=True)


def get_indices_in_folder(glob_dir, idxs=None):
    """Get indices of items in a glob_dir
    optionally: enforce that they are in the list idxs
    """
    idxs_available = []
    for f in glob_dir:
        if "log" in f:
            continue
        idx = int(f.split("/")[-1].replace(".txt", ""))
        if idxs is not None:
            try:
                iterator = iter(idxs)
            except TypeError:
                # not iterable
                if not (idx == idxs):
                    continue
            else:
                # iterable
                if idx not in idxs:
                    continue
        if idx not in idxs_available:
            idxs_available.append(idx)
    return idxs_available


def color_from_object_type(det_type, no_white=False, no_black=False):
    if det_type == "detection":
        cstring = "green" if no_black else "black"
    elif det_type == "truth":
        cstring = "cyan" if no_white else "white"
    elif det_type == "false_negative":
        cstring = "yellow"
    elif det_type == "false_positive":
        cstring = "red"
    elif det_type == "true_positive":
        cstring = "blue"
    elif det_type == "dontcare":
        cstring = "brown"
    else:
        raise NotImplementedError(f"{det_type} not available for color")
    return parse_color_string(cstring)


def parse_color_string(cstring):
    if cstring == "white":
        lcolor = (255, 255, 255)
    elif cstring == "green":
        lcolor = (0, 255, 0)
    elif cstring == "red":
        lcolor = (255, 0, 0)
    elif cstring == "blue":
        lcolor = (0, 0, 255)
    elif cstring == "cyan":
        lcolor = (0, 255, 255)
    elif cstring == "lightblue":
        lcolor = (51, 255, 255)
    elif cstring == "black":
        lcolor = (0, 0, 0)
    elif cstring == "yellow":
        lcolor = (236, 213, 64)
        # lcolor = (255, 255, 0)
    elif cstring == "brown":
        lcolor = (165, 42, 42)
    else:
        raise ValueError(f"Unknown color type {cstring}")
    return lcolor
