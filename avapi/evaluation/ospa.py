# Optimal subpattern assignment metric for object tracking

import numpy as np
from avstack.modules.assignment import gnn_single_frame_assign


class OspaMetric:
    @staticmethod
    def _cost(list_shorter: list, list_longer: list, p: float = 1.0, c: float = 1.0):
        n = len(list_longer)
        m = len(list_shorter)
        A = np.array(
            [[np.linalg.norm(l1 - l2) for l1 in list_shorter] for l2 in list_longer]
        )
        assign = gnn_single_frame_assign(A, cost_threshold=c)
        rows, cols = assign.rows_and_cols()
        c_una = c * len(assign.unassigned_rows)
        distance = (
            1 / n * (np.sum(A[rows, cols]) + c_una) ** p + (n - m) * c**p
        ) ** 1 / p
        return distance

    @staticmethod
    def cost(tracks: list, truths: list, p: float = 1.0, c: float = 1.0):
        if len(tracks) <= len(truths):
            return OspaMetric._cost(tracks, truths, p=p, c=c)
        else:
            return OspaMetric._cost(truths, tracks, p=p, c=c)
