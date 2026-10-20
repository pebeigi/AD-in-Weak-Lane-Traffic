# -*- coding: utf-8 -*-
"""
Frame construction for SDPM calibration and paired evaluation.

Adds vehicle length / width columns and neighbour tuples with bounding-box
sizes so interaction forces use realistic dimensions.
"""

from __future__ import annotations

import math

from rau_calibration import NUM_NEAREST_NEIGHBORS, SAMPLE_STEP


def build_frame_data_with_dimensions(
    df,
    num_nearest: int = NUM_NEAREST_NEIGHBORS,
    sample_step: int = SAMPLE_STEP,
):
    """Same semantics as ``rau_calibration.build_frame_data``, plus:

      - keys ``length_m``, ``width_m`` per ego row;
      - each neighbour tuple
        ``(nx, ny, vx, vy, length_m_nb, width_m_nb)``.
    """
    df = df.copy()
    if "length_smoothed" not in df.columns:
        df["length_smoothed"] = 4.5
    if "width_smoothed" not in df.columns:
        df["width_smoothed"] = 1.7

    df = df.sort_values(["object_ID", "time"], kind="mergesort").reset_index(drop=True)
    id_last = (
        df.groupby("object_ID", sort=False).agg({"x_m": "last", "y_m": "last"}).to_dict("index")
    )
    id_to_dest = {oid: (id_last[oid]["x_m"], id_last[oid]["y_m"]) for oid in id_last}

    frame_data: dict = {}
    grouped = df.groupby("time")
    times_sorted = sorted(grouped.groups.keys())
    sampled = times_sorted[::sample_step]
    for t in sampled:
        g = grouped.get_group(t)
        user_list = []
        for _, row in g.iterrows():
            oid = row["object_ID"]
            dest_x, dest_y = id_to_dest.get(oid, (row["x_m"], row["y_m"]))
            user_list.append(
                {
                    "id": oid,
                    "x": row["x_m"],
                    "y": row["y_m"],
                    "vx": row["vx"],
                    "vy": row["vy"],
                    "dest_x": dest_x,
                    "dest_y": dest_y,
                    "lane_kf": float(row["lane_kf"]),
                    "run_id": float(row["run_id"]),
                    "length_m": float(row["length_smoothed"]),
                    "width_m": float(row["width_smoothed"]),
                    "neighbors": [],
                }
            )
        for i in range(len(user_list)):
            ui = user_list[i]
            px_i, py_i = ui["x"], ui["y"]
            ri, li = ui["run_id"], ui["lane_kf"]
            dists = []
            for j in range(len(user_list)):
                if i == j:
                    continue
                uj = user_list[j]
                if uj["run_id"] != ri or uj["lane_kf"] != li:
                    continue
                d = math.hypot(px_i - uj["x"], py_i - uj["y"])
                dists.append((d, j))
            dists.sort(key=lambda x: x[0])
            nbs = []
            for _, j in dists[:num_nearest]:
                u = user_list[j]
                nbs.append(
                    (
                        u["x"],
                        u["y"],
                        u["vx"],
                        u["vy"],
                        float(u["length_m"]),
                        float(u["width_m"]),
                    )
                )
            user_list[i]["neighbors"] = nbs
        frame_data[t] = user_list
    return frame_data
