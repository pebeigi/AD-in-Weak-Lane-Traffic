"""Deterministic seeds from the canonical trajectory key.

Python's built-in hash() is randomized per process, so it is not used.
The digest is SHA-256 of a fixed ASCII string.
"""

from __future__ import annotations

import hashlib
import random

GLOBAL_EXPERIMENT_SEED = 20261004


def execution_seed(run_id, lane_kf, vehicle_id, stream: str) -> int:
    """32-bit seed for one stochastic stream of one physical trajectory."""
    payload = (
        f"{GLOBAL_EXPERIMENT_SEED}|{int(run_id)}|{int(lane_kf)}|"
        f"{int(vehicle_id)}|{stream}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def execution_rng(run_id, lane_kf, vehicle_id, stream: str) -> random.Random:
    return random.Random(execution_seed(run_id, lane_kf, vehicle_id, stream))


def frame_key(frame_data, focus_id):
    times = sorted(frame_data)
    if not times:
        raise ValueError("empty frames")
    matches = [u for u in frame_data[times[0]] if int(u["id"]) == int(focus_id)]
    if not matches:
        raise KeyError(focus_id)
    u0 = matches[0]
    return int(u0["run_id"]), int(u0.get("lane_kf")), int(focus_id)
