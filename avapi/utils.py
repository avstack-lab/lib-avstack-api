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
