# -*- coding: utf-8 -*-
"""
Risk-aware utility (RAU) model calibration.

New input data: `Final_Lebanon_Data.csv` with columns:
  - id, time, xloc_kf, yloc_kf, lane_kf, speed_kf, acceleration_kf, length_smoothed, width_smoothed, class, run_id

We calibrate cars separately for each combination of:
  - run_id (run_index)
  - lane_kf (direction: east->west vs west->east)

Within each (run_id, lane_kf) subset we run GA calibration per `id` and save results into:
  - ./rau_calibration_outputs/run_{run_id}_direction_{lane_kf}/

The same trajectory figure is also mirrored to ./comparison_visualizations/ with names like
run_{run_id}_direction_{lane_kf}_id_{id}_RAU.png (disable with `--no-comparison-plots`).

For a quick check, use coarser frames and a smaller parallel search:
  python rau_calibration.py --fast --workers 6
  python rau_calibration.py --fast --workers 8 --max-vehicles 2 --sample-step 5
"""

import numpy as np
import pandas as pd
import math
import random
import os
import sys
import argparse
import contextlib
from multiprocessing import get_context

# Non-interactive backend: avoids Qt in spawn workers (Windows) and cleaner Ctrl+C.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from highway_geometry import (
    clip_position_to_lane_envelope,
    envelope_lower_upper_polylines,
    lane_boundary_polylines_m,
    load_envelope_boundaries,
)

from comparison_exports import (
    resolve_comparison_output_dir,
    save_figure_to_comparison,
)

##############################################################################
# 1) GLOBAL SETTINGS
##############################################################################

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_SCRIPT_DIR, "Final_Lebanon_Data.csv")

NUM_NEAREST_NEIGHBORS = 4
SAMPLE_STEP = 5
MIN_PRESENCE = 600

POP_SIZE = 10
GENERATIONS = 20
MUTATION_RATE = 0.2
ELITISM_COUNT = 4
# Closed-loop early stop on ATE (m) — R² of step changes is rarely satisfied on noisy real data.
ATE_STOP_M = 2.0

OUTPUT_DIR = "./rau_calibration_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loss = WEIGHT_POSITION * SSE_pos / SS_obs_pos + WEIGHT_SPEED * SSE_speed / SS_obs_speed
# (per-axis, normalized; velocity not drowned by position-step magnitudes).
WEIGHT_POSITION = 1.0
WEIGHT_SPEED = 0.5

# RAU anticipation horizon (s) and collision threshold (m), decoupled from dt.
T_ANTICIPATION = 2.0       # seconds ahead for collision evaluation (highway: 2–3 s)
COLLISION_THRESHOLD = 4.0  # min safe distance (m); car length + buffer
ACCEL_PENALTY_COEF = 0.5   # penalty for harsh acceleration in utility

# Static infrastructure: distance to south/median/north polylines (see highway_geometry.py)
# Margin inside the percentile boundary at which risk rises. The corridor is
# about 8–10 m wide, so a 4 m threshold would mark the centerline itself as risky.
BOUNDARY_THRESHOLD_M = 1.0
# Default / fallback when pvec has no W_boundary (e.g. old 5-param CSVs)
DEFAULT_BOUNDARY_COLLISION_WEIGHT = 800.0

_ERF = np.vectorize(math.erf, otypes=[float])


def _erf_uv(z):
    return np.asarray(_ERF(np.asarray(z, dtype=float)), dtype=float)


def _boundary_prob_grid(ex, ey, run_id, lane_kf, sigma_b):
    from highway_geometry import boundary_prob_array

    return boundary_prob_array(ex, ey, run_id, lane_kf, BOUNDARY_THRESHOLD_M, sigma_b)

##############################################################################
# 2) LOAD AND PREPROCESS DATA
##############################################################################

def load_trajectory_data(csv_path):
    """Load `Final_Lebanon_Data.csv` and map columns to internal names:
    object_ID = id, x_m = xloc_kf, y_m = yloc_kf, time = time.
    Keeps lane_kf and run_id for later subset calibration.
    """
    df = pd.read_csv(csv_path)

    # Required columns in the new dataset
    required = ["id", "time", "xloc_kf", "yloc_kf", "lane_kf", "run_id", "class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Numeric coercion
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).copy()

    # Internal names expected by the rest of this script
    df["object_ID"] = df["id"]
    # xloc_kf / yloc_kf are already in meters in the new dataset
    df["x_m"] = df["xloc_kf"]
    df["y_m"] = df["yloc_kf"]

    # Sort for deterministic velocity computations and time grouping
    df = df.sort_values(by=["run_id", "lane_kf", "object_ID", "time"]).reset_index(drop=True)
    return df

##############################################################################
# 3) VELOCITIES FROM POSITION DIFFERENCES (per object_ID)
##############################################################################

def compute_velocity_components(df):
    """Add vx, vy (m/s) per row using finite difference per object_ID."""
    df = df.copy()
    df["vx"] = np.nan
    df["vy"] = np.nan
    for oid, g in df.groupby("object_ID"):
        g = g.sort_values("time")
        t = g["time"].values.astype(float)
        x = g["x_m"].values.astype(float)
        y = g["y_m"].values.astype(float)
        dt = np.diff(t)
        dx = np.diff(x)
        dy = np.diff(y)
        vx = np.zeros_like(x)
        vy = np.zeros_like(y)
        vx[1:] = dx / (dt + 1e-9)
        vy[1:] = dy / (dt + 1e-9)
        if len(dt) > 0:
            vx[0] = vx[1]
            vy[0] = vy[1]
        idx = df["object_ID"] == oid
        df.loc[idx, "vx"] = vx
        df.loc[idx, "vy"] = vy
    return df

##############################################################################
# 4) RAU PARAMETER VECTOR (two risk weights and four utility/noise parameters)
#    [Wc_cc, eta, xi, tau, v_pref, W_boundary]
##############################################################################

def random_rau_parameters():
    return [
        random.uniform(100, 1000),   # Wc_cc (car-car collision weight)
        random.uniform(1, 5),       # eta
        random.uniform(0.5, 10),    # xi
        random.uniform(0.2, 1.5),   # tau
        random.uniform(5, 35),      # v_pref (m/s, e.g. 18–35 m/s for highway)
        random.uniform(10, 200),  # W_boundary (highway envelope penalty; same scale as Wc_cc)
    ]

def clamp_rau_parameters(p):
    p[0] = min(max(p[0], 10), 5000)
    p[1] = min(max(p[1], 0.1), 10)
    p[2] = min(max(p[2], 0.1), 15)
    p[3] = min(max(p[3], 0.1), 3.0)
    p[4] = min(max(p[4], 1), 50)
    p[5] = min(max(p[5], 0.0), 10000.0)
    return p

def mutate_rau_parameters(p, rate=0.2):
    for i in range(len(p)):
        if random.random() < rate:
            if i == 0:
                p[i] += random.uniform(-100, 100)
            elif i in [1, 2]:
                p[i] += random.uniform(-0.5, 0.5)
            elif i == 3:
                p[i] += random.uniform(-0.2, 0.2)
            elif i == 5:
                p[i] += random.uniform(-150, 150)
            else:
                p[i] += random.uniform(-2, 2)
    return clamp_rau_parameters(p)

def crossover_rau_parameters(p1, p2):
    idx = random.randint(1, len(p1) - 1)
    c1 = p1[:idx] + p2[idx:]
    c2 = p2[:idx] + p1[idx:]
    return clamp_rau_parameters(c1), clamp_rau_parameters(c2)

##############################################################################
# 5) RAU CHOICE — anticipation horizon, speed-based sigma, acceleration penalty
##############################################################################

def select_rau_acceleration(
    px, py, vx, vy, pvec, neighbors, dest_x, dest_y, dt=0.1, lane_kf=None, run_id=None
):
    """Decoupled T_anticipation from dt to avoid myopic risk. Search over acceleration + angle. Returns (ax, ay)."""
    if len(pvec) < 6:
        pvec = list(pvec) + [DEFAULT_BOUNDARY_COLLISION_WEIGHT] * (6 - len(pvec))
    Wc_cc, eta, xi, tau, v_pref, w_boundary = pvec
    curr_speed = math.hypot(vx, vy)
    dx_dest = dest_x - px
    dy_dest = dest_y - py
    dist_to_dest = math.hypot(dx_dest, dy_dest)
    if dist_to_dest < 1e-6:
        goal_theta = math.atan2(vy, vx) if curr_speed >= 1e-9 else 0.0
    else:
        goal_theta = math.atan2(dy_dest, dx_dest)

    # Finer action grid to reduce quantization artifacts in trend following
    # (e.g., observed deceleration while chosen action still accelerates).
    accel_candidates = np.linspace(-4.0, 2.0, 20)
    angle_candidates = np.linspace(
        goal_theta - math.radians(15), goal_theta + math.radians(15), 10
    )

    best_v = curr_speed
    best_th = goal_theta

    use_boundary = lane_boundary_polylines_m(run_id, lane_kf) is not None
    # Same 20 x 10 grid as the previous scalar loop. Evaluated as arrays so the
    # argmax is identical and a rollout is cheap enough to calibrate.
    acc = np.asarray(accel_candidates, dtype=float)
    ang = np.asarray(angle_candidates, dtype=float)
    aa, gg = np.meshgrid(acc, ang, indexing="ij")
    sp = np.maximum(0.0, curr_speed + aa * dt)
    align = np.maximum(0.0, np.cos(gg - goal_theta))
    if v_pref <= 1e-9:
        val = np.zeros_like(sp)
    else:
        ratio = sp / v_pref
        val = (eta * align) * (ratio / np.power(1.0 + ratio, xi / 2.0))
    if neighbors:
        nb = np.asarray(neighbors, dtype=float)
        if nb.ndim == 1:
            nb = nb.reshape(1, -1)
        ex = px + sp * np.cos(gg) * T_ANTICIPATION
        ey = py + sp * np.sin(gg) * T_ANTICIPATION
        exn = nb[:, 0] + nb[:, 2] * T_ANTICIPATION
        eyn = nb[:, 1] + nb[:, 3] * T_ANTICIPATION
        dist = np.hypot(ex[..., None] - exn, ey[..., None] - eyn)
        sigma = 2.0 + 0.1 * sp
        z = (dist - COLLISION_THRESHOLD) / sigma[..., None]
        p_each = 0.5 * (1.0 - _erf_uv(z / math.sqrt(2.0)))
        p_cc = np.max(p_each, axis=-1)
        w_c = float(Wc_cc)
    else:
        p_cc = np.zeros_like(sp)
        w_c = 0.0
    w_b = float(w_boundary) if use_boundary else 0.0
    if w_b <= 0.0:
        utility = (1.0 - p_cc) * val - (p_cc * w_c)
    else:
        ex = px + sp * np.cos(gg) * T_ANTICIPATION
        ey = py + sp * np.sin(gg) * T_ANTICIPATION
        sigma_b = 2.0 + 0.1 * sp
        p_b = _boundary_prob_grid(ex, ey, run_id, lane_kf, sigma_b)
        utility = (1.0 - p_cc) * (1.0 - p_b) * val - (p_cc * w_c) - (p_b * w_b)
    utility = utility - ACCEL_PENALTY_COEF * (aa ** 2)
    flat = int(np.argmax(utility))
    best_v = float(sp.ravel()[flat])
    best_th = float(gg.ravel()[flat])

    vx_fin = best_v * math.cos(best_th)
    vy_fin = best_v * math.sin(best_th)
    ax = (vx_fin - vx) / dt
    ay = (vy_fin - vy) / dt
    return ax, ay

def rau_acceleration(
    px, py, vx, vy, pvec, neighbors, dest_x, dest_y, dt=0.1, lane_kf=None, run_id=None
):
    return select_rau_acceleration(
        px, py, vx, vy, pvec, neighbors, dest_x, dest_y, dt, lane_kf=lane_kf, run_id=run_id
    )

##############################################################################
# 6) SINGLE-AGENT SIMULATION (car)
##############################################################################

def simulate_rau_single_agent(
    px,
    py,
    vx,
    vy,
    pvec,
    neighbors,
    dest_x,
    dest_y,
    total_dt=1.0,
    substeps=5,
    lane_kf=None,
    run_id=None,
    clip_to_envelope=True,
    envelope_clip_margin_m=0.35,
    clip_initial=True,
    rng=None,
):
    """
    Advance one RAU agent through one or more integration substeps.
    If clip_to_envelope: project (x,y) into the L/U lane envelope after spawn and each substep
    (same as multi-agent sim). If clip_to_envelope and clip_initial is False, the pre-step snap
    is skipped so the first position matches a logged state (e.g. calibration at GPS); substeps
    can still snap when clip_to_envelope is on.
    """
    h = total_dt / substeps
    tau = pvec[3]
    x, y, vx_, vy_ = px, py, vx, vy
    # One generator for this call. Callers that pass rng keep a single stream
    # across the trip, so the same vehicle and parameters always replay the same noise.
    if rng is None:
        rng = random.Random(0)
    if clip_to_envelope and clip_initial:
        x, y, _ = clip_position_to_lane_envelope(
            x, y, run_id, lane_kf, margin_m=envelope_clip_margin_m
        )
    wx, wy = 0.0, 0.0
    for _ in range(substeps):
        nb_list = [(n[0], n[1], n[2], n[3]) for n in neighbors]
        from highway_geometry import goal_ahead

        g = goal_ahead(x, y, run_id, lane_kf)
        if g is not None:
            dest_x, dest_y = g
        ax, ay = rau_acceleration(
            x, y, vx_, vy_, pvec, nb_list, dest_x, dest_y, dt=h, lane_kf=lane_kf, run_id=run_id
        )
        wx = wx * (1.0 - h / tau) + math.sqrt(h) * rng.gauss(0, 1)
        wy = wy * (1.0 - h / tau) + math.sqrt(h) * rng.gauss(0, 1)
        sigma_v = 0.1
        vx_ = vx_ + ax * h + sigma_v * wx
        vy_ = vy_ + ay * h + sigma_v * wy
        if not math.isfinite(vx_):
            vx_ = 0.0
        if not math.isfinite(vy_):
            vy_ = 0.0
        sp = math.hypot(vx_, vy_)
        if sp > 45.0:
            vx_ *= 45.0 / sp
            vy_ *= 45.0 / sp
        x += vx_ * h
        y += vy_ * h
        if not (math.isfinite(x) and math.isfinite(y)):
            x, y = px, py
        if clip_to_envelope:
            x, y, _ = clip_position_to_lane_envelope(
                x, y, run_id, lane_kf, margin_m=envelope_clip_margin_m
            )
        # Update neighbor positions for next substep (simple Euler).
        # Keep length and width when the caller passed a 6-tuple.
        advanced = []
        for n in neighbors:
            nx, ny, nvx, nvy = n[0], n[1], n[2], n[3]
            head = (nx + nvx * h, ny + nvy * h, nvx, nvy)
            advanced.append(head + tuple(n[4:]) if len(n) > 4 else head)
        neighbors = advanced
    return x, y, vx_, vy_

##############################################################################
# 7) BUILD FRAME DATA (per time, 8 nearest neighbors per vehicle)
##############################################################################

def build_frame_data(df, num_nearest=NUM_NEAREST_NEIGHBORS, sample_step=SAMPLE_STEP):
    # Per-ID destination = last (x_m, y_m) for that ID in df (must be time-ordered: pandas "last"
    # is the last row in the group, rather than max(time).
    df = df.sort_values(["object_ID", "time"], kind="mergesort").reset_index(drop=True)
    id_last = df.groupby("object_ID", sort=False).agg({"x_m": "last", "y_m": "last"}).to_dict("index")
    id_to_dest = {oid: (id_last[oid]["x_m"], id_last[oid]["y_m"]) for oid in id_last}

    frame_data = {}
    grouped = df.groupby("time")
    times_sorted = sorted(grouped.groups.keys())
    sampled = times_sorted[::sample_step]
    for t in sampled:
        g = grouped.get_group(t)
        user_list = []
        for _, row in g.iterrows():
            oid = row["object_ID"]
            dest_x, dest_y = id_to_dest.get(oid, (row["x_m"], row["y_m"]))
            user_list.append({
                "id": oid,
                "x": row["x_m"],
                "y": row["y_m"],
                "vx": row["vx"],
                "vy": row["vy"],
                "dest_x": dest_x,
                "dest_y": dest_y,
                "lane_kf": float(row["lane_kf"]),
                "run_id": float(row["run_id"]),
                "neighbors": [],
            })
        # For each vehicle, set neighbors = 8 nearest among vehicles in the SAME
        # (run_id, lane_kf) — same dataset and same direction. Calibration always
        # passes a pre-filtered df, but enforce here too so cross-direction or
        # cross-run vehicles can never leak into neighbor lists.
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
                nbs.append((u["x"], u["y"], u["vx"], u["vy"]))
            user_list[i]["neighbors"] = nbs
        frame_data[t] = user_list
    return frame_data

##############################################################################
# 8) METRICS
##############################################################################

def compute_rau_metrics(
    frame_data,
    pvec,
    focus_id,
    clip_to_envelope=True,
    envelope_clip_margin_m=0.35,
    rng=None,
):
    """Evaluate metrics for focus_id with a **whole-trip (closed-loop) roll-out**: ego starts
    once from ground truth at the first time; for each following segment, integration continues
    from the **simulated** end state, while neighbors, destination, and lane context are taken
    from the **logged** frame at the segment start. If a segment is skipped (missing id), ego
    is re-synced to ground truth at t2 (or t1) so the chain stays defined across gaps.

    R² uses step changes (Δx, Δy) vs logged step, with predicted step from the rolled trajectory.
    ADE / ATE / FDE: planar error at each segment end (pred vs obs at t2), last = FDE.

    Speed term: separate squared errors for ``vx`` and ``vy`` at each segment end (sum over time of
    ``(vx_sim - vx_obs)²`` and ``(vy_sim - vy_obs)²``). We do **not** compare scalar speed magnitudes
    ``hypot(vx,vy)``: that could hide opposite axis errors when total speed still matches observed.
    """
    times_list = sorted(frame_data.keys())
    bad = {
        "MSE": 1e9,
        "RMSE": 1e9,
        "MAE": 1e9,
        "MAPE": 1e9,
        "NRMSE": 1e9,
        "SSE": 1e9,
        "R-squared": -999,
        # GA maximizes Error = 1/(1+L) in (0, 1]. A failed rollout must lose, not win.
        "Error": 1e-12,
        "ATE": 1e9,
        "ADE": 1e9,
        "FDE": 1e9,
    }
    if len(times_list) < 2:
        return bad
    time_map = {t: {u["id"]: u for u in frame_data[t]} for t in times_list}
    t0 = times_list[0]
    if focus_id not in time_map[t0]:
        return bad
    u0 = time_map[t0][focus_id]
    x, y, vx, vy = u0["x"], u0["y"], u0["vx"], u0["vy"]
    if rng is None:
        from analysis.trajectory_seed import execution_rng

        rng = execution_rng(u0.get("run_id"), u0.get("lane_kf"), focus_id, "rau")
    resynced = False
    step_err_dx, step_err_dy = [], []
    step_obs_dx, step_obs_dy = [], []
    speed_err_vx, speed_err_vy = [], []
    speed_obs_vx, speed_obs_vy = [], []
    disp_err_list = []
    sim_x, sim_y, obs_x, obs_y = [], [], [], []
    sim_speed, obs_speed = [], []
    for i in range(len(times_list) - 1):
        t1, t2 = times_list[i], times_list[i + 1]
        dt = t2 - t1
        if dt <= 0:
            continue
        id_map1, id_map2 = time_map[t1], time_map[t2]
        if focus_id not in id_map1 or focus_id not in id_map2:
            if focus_id in id_map2:
                ur = id_map2[focus_id]
                x, y, vx, vy = ur["x"], ur["y"], ur["vx"], ur["vy"]
            elif focus_id in id_map1:
                ur = id_map1[focus_id]
                x, y, vx, vy = ur["x"], ur["y"], ur["vx"], ur["vy"]
            resynced = True
            continue
        u1, u2 = id_map1[focus_id], id_map2[focus_id]
        px1, py1 = u1["x"], u1["y"]
        nb1 = u1["neighbors"]
        px2_obs, py2_obs = u2["x"], u2["y"]
        vx2_obs, vy2_obs = u2["vx"], u2["vy"]
        speed_obs_vx.append(vx2_obs)
        speed_obs_vy.append(vy2_obs)
        prev_x, prev_y = x, y
        lk = u1.get("lane_kf")
        rid = u1.get("run_id")
        from_logged = (i == 0) or resynced
        resynced = False
        # Snap only when continuing from a simulated (possibly off-lane) state, not at GPS.
        clip_initial = not from_logged
        pxp, pyp, vxp, vyp = simulate_rau_single_agent(
            x, y, vx, vy, pvec, nb1, u1["dest_x"], u1["dest_y"],
            total_dt=dt,
            substeps=5,
            lane_kf=lk,
            run_id=rid,
            clip_to_envelope=clip_to_envelope,
            envelope_clip_margin_m=envelope_clip_margin_m,
            clip_initial=clip_initial,
            rng=rng,
        )
        # Step: predicted vs observed (logged u1->u2)
        dx_obs = px2_obs - px1
        dy_obs = py2_obs - py1
        dx_pred = pxp - prev_x
        dy_pred = pyp - prev_y
        step_err_dx.append(dx_pred - dx_obs)
        step_err_dy.append(dy_pred - dy_obs)
        step_obs_dx.append(dx_obs)
        step_obs_dy.append(dy_obs)
        speed_err_vx.append(vxp - vx2_obs)
        speed_err_vy.append(vyp - vy2_obs)
        disp_err_list.append(math.hypot(pxp - px2_obs, pyp - py2_obs))
        sim_x.append(pxp)
        sim_y.append(pyp)
        obs_x.append(px2_obs)
        obs_y.append(py2_obs)
        sim_speed.append(math.hypot(vxp, vyp))
        obs_speed.append(math.hypot(vx2_obs, vy2_obs))
        x, y, vx, vy = pxp, pyp, vxp, vyp
    step_err_dx = np.array(step_err_dx)
    step_err_dy = np.array(step_err_dy)
    step_obs_dx = np.array(step_obs_dx)
    step_obs_dy = np.array(step_obs_dy)
    speed_err_vx = np.array(speed_err_vx)
    speed_err_vy = np.array(speed_err_vy)
    n = len(step_err_dx)
    if n == 0:
        return bad
    # Position SSE (step changes): sum of squared errors on Δx and Δy steps (per-axis, not |Δp|).
    SSE_position = float(np.sum(step_err_dx ** 2 + step_err_dy ** 2))
    # Velocity: per-component SSE (vx then vy), summed — not (||v_sim|| − ||v_obs||)².
    SSE_speed_vx = float(np.sum(speed_err_vx ** 2))
    SSE_speed_vy = float(np.sum(speed_err_vy ** 2))
    SSE_speed = SSE_speed_vx + SSE_speed_vy
    # Normalize each term by the corresponding observed energy so highway-scale step magnitudes
    # don't drown velocity errors. Each ratio is unitless; ~0 = perfect, ~1 = useless.
    SS_obs_position = float(np.sum(step_obs_dx ** 2 + step_obs_dy ** 2)) + 1e-9
    SS_obs_speed = (
        float(np.sum(np.asarray(speed_obs_vx, dtype=float) ** 2))
        + float(np.sum(np.asarray(speed_obs_vy, dtype=float) ** 2))
        + 1e-9
    )
    norm_pos = SSE_position / SS_obs_position
    norm_speed = SSE_speed / SS_obs_speed
    # weighted_MSE keeps its name (used by error_history) but is now a unitless combined loss.
    weighted_MSE = WEIGHT_POSITION * norm_pos + WEIGHT_SPEED * norm_speed
    # R² and position metrics unchanged (on position steps)
    SSE = SSE_position
    mean_dx = np.mean(step_obs_dx)
    mean_dy = np.mean(step_obs_dy)
    SStot = np.sum((step_obs_dx - mean_dx) ** 2 + (step_obs_dy - mean_dy) ** 2)
    r2 = 1.0 - (SSE / SStot) if SStot > 1e-9 else -999
    MSE = SSE / n
    RMSE = math.sqrt(MSE)
    MAE = np.mean(np.abs(np.concatenate([step_err_dx, step_err_dy])))
    step_obs_norm = np.hypot(step_obs_dx, step_obs_dy)
    step_pred_dx = step_obs_dx + step_err_dx
    step_pred_dy = step_obs_dy + step_err_dy
    step_pred_norm = np.hypot(step_pred_dx, step_pred_dy)
    mask = step_obs_norm > 1e-9
    MAPE = np.mean(np.abs((step_pred_norm[mask] - step_obs_norm[mask]) / step_obs_norm[mask])) * 100 if np.any(mask) else 1e9
    dx_range = step_obs_dx.max() - step_obs_dx.min() if step_obs_dx.size else 1e-9
    dy_range = step_obs_dy.max() - step_obs_dy.min() if step_obs_dy.size else 1e-9
    NRMSE = RMSE / (((dx_range + dy_range) * 0.5) if (dx_range + dy_range) > 0 else 1e-9)
    disp = np.asarray(disp_err_list, dtype=float)
    ADE = float(np.mean(disp))
    ATE = float(np.sqrt(np.mean(disp ** 2)))
    FDE = float(disp[-1])
    return {
        "MSE": MSE, "RMSE": RMSE, "MAE": MAE, "MAPE": MAPE,
        "NRMSE": NRMSE, "SSE": SSE, "R-squared": r2,
        "Error": 1.0 / (1.0 + weighted_MSE),
        "ATE": ATE,
        "ADE": ADE,
        "FDE": FDE,
        "sim_x": sim_x,
        "sim_y": sim_y,
        "obs_x": obs_x,
        "obs_y": obs_y,
        "sim_speed": sim_speed,
        "obs_speed": obs_speed,
    }

##############################################################################
# 9) GENETIC ALGORITHM
##############################################################################

def evaluate_rau_population(
    frame_data,
    population,
    focus_id,
    clip_to_envelope=True,
    envelope_clip_margin_m=0.35,
):
    return [
        compute_rau_metrics(
            frame_data,
            pvec,
            focus_id,
            clip_to_envelope=clip_to_envelope,
            envelope_clip_margin_m=envelope_clip_margin_m,
        )["Error"]
        for pvec in population
    ]


# Multiprocessing: frame_data + focus_id copied once per GA run into worker processes.
_MP_FRAME = None
_MP_FOCUS = None
_MP_CLIP = True
_MP_MARGIN = 0.35


def _mp_init_eval(frame_data, focus_id, clip_to_envelope, envelope_clip_margin_m):
    global _MP_FRAME, _MP_FOCUS, _MP_CLIP, _MP_MARGIN
    _MP_FRAME = frame_data
    _MP_FOCUS = focus_id
    _MP_CLIP = clip_to_envelope
    _MP_MARGIN = envelope_clip_margin_m


def _mp_eval_error(pvec):
    return compute_rau_metrics(
        _MP_FRAME,
        pvec,
        _MP_FOCUS,
        clip_to_envelope=_MP_CLIP,
        envelope_clip_margin_m=_MP_MARGIN,
    )["Error"]


def evaluate_population_parallel(
    frame_data,
    population,
    focus_id,
    n_workers: int,
    clip_to_envelope=True,
    envelope_clip_margin_m=0.35,
):
    """Same fitness as ``evaluate_rau_population``, distributed across processes."""
    if n_workers <= 1:
        return evaluate_rau_population(
            frame_data,
            population,
            focus_id,
            clip_to_envelope=clip_to_envelope,
            envelope_clip_margin_m=envelope_clip_margin_m,
        )
    ctx = get_context("spawn")
    pool = ctx.Pool(
        n_workers,
        initializer=_mp_init_eval,
        initargs=(frame_data, focus_id, clip_to_envelope, envelope_clip_margin_m),
    )
    try:
        return pool.map(_mp_eval_error, population, chunksize=1)
    except KeyboardInterrupt:
        pool.terminate()
        raise
    finally:
        # After terminate(), close() may be a no-op; join() reaps workers (important on Windows).
        with contextlib.suppress(Exception):
            pool.close()
        with contextlib.suppress(Exception):
            pool.join()


def run_rau_calibration(
    frame_data_sub,
    focus_id,
    pop_size=POP_SIZE,
    generations=GENERATIONS,
    ate_stop_m=ATE_STOP_M,
    elitism_count=ELITISM_COUNT,
    n_workers: int = 1,
    clip_to_envelope=True,
    envelope_clip_margin_m=0.35,
    search_seed=None,
):
    elitism = max(1, min(elitism_count, pop_size - 1))
    if search_seed is None:
        from analysis.trajectory_seed import execution_seed, frame_key

        run_id, lane_kf, vid = frame_key(frame_data_sub, focus_id)
        random.seed(execution_seed(run_id, lane_kf, vid, "rau-search"))
    else:
        random.seed(int(search_seed))
    population = [clamp_rau_parameters(random_rau_parameters()) for _ in range(pop_size)]
    best_vec, best_score = None, -1e9
    error_history = []  # weighted_MSE of running best per generation (monotonically non-increasing)
    for gen in range(generations):
        scores = evaluate_population_parallel(
            frame_data_sub,
            population,
            focus_id,
            n_workers,
            clip_to_envelope=clip_to_envelope,
            envelope_clip_margin_m=envelope_clip_margin_m,
        )
        pop_sorted = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        if pop_sorted[0][1] > best_score:
            best_score = pop_sorted[0][1]
            best_vec = pop_sorted[0][0]
        # Convert fitness back to weighted_MSE: Error = 1/(1+weighted_MSE) -> weighted_MSE = 1/Error - 1.
        err_running = (1.0 / best_score - 1.0) if best_score > 1e-12 else float("inf")
        error_history.append(float(err_running))
        curr_metrics = compute_rau_metrics(
            frame_data_sub,
            best_vec,
            focus_id,
            clip_to_envelope=clip_to_envelope,
            envelope_clip_margin_m=envelope_clip_margin_m,
        )
        if curr_metrics["ATE"] <= ate_stop_m:
            print(f"Early stop gen {gen+1}: ATE = {curr_metrics['ATE']:.2f} m")
            break
        new_pop = [pop_sorted[i][0] for i in range(elitism)]
        while len(new_pop) < pop_size:
            # Tournament, not a uniform draw. A uniform draw keeps mating
            # failed parameter vectors and leaves the search seed-dependent.
            k = min(3, len(pop_sorted))
            p1 = max(random.sample(pop_sorted, k), key=lambda z: z[1])[0]
            p2 = max(random.sample(pop_sorted, k), key=lambda z: z[1])[0]
            c1, c2 = crossover_rau_parameters(p1, p2)
            new_pop.append(mutate_rau_parameters(c1, MUTATION_RATE))
            if len(new_pop) < pop_size:
                new_pop.append(mutate_rau_parameters(c2, MUTATION_RATE))
        population = new_pop[:pop_size]
        if (gen + 1) % 10 == 0:
            print(f"GA gen={gen+1}, best fitness={best_score:.5f}")
    final_metrics = compute_rau_metrics(
        frame_data_sub,
        best_vec,
        focus_id,
        clip_to_envelope=clip_to_envelope,
        envelope_clip_margin_m=envelope_clip_margin_m,
    )
    return best_vec, best_score, final_metrics, error_history

##############################################################################
# 10) PREDICTIONS FOR ONE VEHICLE
##############################################################################

def get_rau_predictions(
    frame_data_sub,
    params,
    focus_id,
    clip_to_envelope=True,
    envelope_clip_margin_m=0.35,
):
    """One row per segment end t2, matching the whole-trip RAU rollout."""
    times_list = sorted(frame_data_sub.keys())
    if len(times_list) < 2:
        return pd.DataFrame([])
    time_dict = {t: {u["id"]: u for u in frame_data_sub[t]} for t in times_list}
    if focus_id not in time_dict[times_list[0]]:
        return pd.DataFrame([])
    t0 = times_list[0]
    u0 = time_dict[t0][focus_id]
    from analysis.trajectory_seed import execution_rng

    rng = execution_rng(u0.get("run_id"), u0.get("lane_kf"), focus_id, "rau")
    x, y, vx, vy = u0["x"], u0["y"], u0["vx"], u0["vy"]
    rows = [
        {
            "id": focus_id,
            "time": t0,
            "x_obs": u0["x"],
            "y_obs": u0["y"],
            "vx_obs": u0["vx"],
            "vy_obs": u0["vy"],
            "x_pred": u0["x"],
            "y_pred": u0["y"],
            "vx_pred": u0["vx"],
            "vy_pred": u0["vy"],
        }
    ]
    resynced = False
    for i in range(len(times_list) - 1):
        t1, t2 = times_list[i], times_list[i + 1]
        dt = t2 - t1
        if dt <= 0:
            continue
        id_map1, id_map2 = time_dict[t1], time_dict[t2]
        if focus_id not in id_map1 or focus_id not in id_map2:
            if focus_id in id_map2:
                ur = id_map2[focus_id]
                x, y, vx, vy = ur["x"], ur["y"], ur["vx"], ur["vy"]
            elif focus_id in id_map1:
                ur = id_map1[focus_id]
                x, y, vx, vy = ur["x"], ur["y"], ur["vx"], ur["vy"]
            resynced = True
            continue
        u1, u2 = id_map1[focus_id], id_map2[focus_id]
        nb1 = u1["neighbors"]
        x2_obs, y2_obs = u2["x"], u2["y"]
        lk = u1.get("lane_kf")
        rid = u1.get("run_id")
        from_logged = (i == 0) or resynced
        resynced = False
        clip_initial = not from_logged
        pxp, pyp, vxp, vyp = simulate_rau_single_agent(
            x, y, vx, vy, params, nb1, u1["dest_x"], u1["dest_y"],
            total_dt=dt,
            substeps=5,
            lane_kf=lk,
            run_id=rid,
            clip_to_envelope=clip_to_envelope,
            envelope_clip_margin_m=envelope_clip_margin_m,
            clip_initial=clip_initial,
            rng=rng,
        )
        rows.append({
            "id": focus_id, "time": t2,
            "x_obs": x2_obs, "y_obs": y2_obs,
            "vx_obs": u2["vx"], "vy_obs": u2["vy"],
            "x_pred": pxp, "y_pred": pyp,
            "vx_pred": vxp, "vy_pred": vyp,
        })
        x, y, vx, vy = pxp, pyp, vxp, vyp
    return pd.DataFrame(rows)

##############################################################################
# 11) SAVE PLOT: 2x3 subplots — observed vs RAU
##############################################################################

def save_rau_trajectory_plot(
    df_pred,
    agent_id,
    output_dir,
    run_id=None,
    lane_kf=None,
    error_history=None,
    comparison_dir=None,
    comparison_model_slug="RAU",
):
    """One figure, 2 rows x 3 cols:
    Row 0: xy trajectory | time vs x | time vs y
    Row 1: time vs vx    | time vs vy | GA generation vs running-best error (weighted_MSE).
    """
    if df_pred.empty or len(df_pred) < 2:
        return

    def _plot_with_uncertainty_band(ax, t, obs, pred, obs_label, pred_label, y_label, title):
        t = np.asarray(t, dtype=float)
        obs = np.asarray(obs, dtype=float)
        pred = np.asarray(pred, dtype=float)
        if t.size < 2:
            return
        order = np.argsort(t)
        t = t[order]
        obs = obs[order]
        pred = pred[order]

        abs_err = np.abs(obs - pred)
        win = max(3, int(t.size // 10))
        if win % 2 == 0:
            win += 1
        # Robust local band from rolling q80 absolute error.
        err_local = (
            pd.Series(abs_err)
            .rolling(window=win, min_periods=1, center=True)
            .quantile(0.7)
            .to_numpy()
        )
        err_floor = float(np.nanquantile(abs_err, 0.15)) if np.isfinite(abs_err).any() else 0.0
        band = np.maximum(err_local, err_floor)

        ax.fill_between(
            t,
            pred - band,
            pred + band,
            color="r",
            alpha=0.18,
            label="RAU uncertainty band",
            zorder=1,
        )
        ax.plot(t, obs, "b-o", label=obs_label, markersize=4, zorder=3)
        ax.plot(t, pred, "r-x", label=pred_label, markersize=4, zorder=4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # (0,0) xy: observed vs RAU trajectory (+ envelope)
    ax_xy = axes[0, 0]
    polys = envelope_lower_upper_polylines(run_id, lane_kf)
    if polys is not None:
        lower, upper = polys
        ax_xy.plot(
            lower[:, 0], lower[:, 1], color="0.45", ls="-", lw=1.0, alpha=0.75,
            label="Envelope lower", zorder=0,
        )
        ax_xy.plot(
            upper[:, 0], upper[:, 1], color="0.45", ls="--", lw=1.0, alpha=0.75,
            label="Envelope upper", zorder=0,
        )
    ax_xy.plot(df_pred["x_obs"], df_pred["y_obs"], "b-o", label="Actual", markersize=4, zorder=2)
    ax_xy.plot(df_pred["x_pred"], df_pred["y_pred"], "r-x", label="RAU simulated", markersize=4, zorder=2)
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("xy trajectory")
    ax_xy.legend(loc="best", fontsize=8)
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.3)
    # (0,1) time vs x
    ax_tx = axes[0, 1]
    _plot_with_uncertainty_band(
        ax_tx,
        df_pred["time"],
        df_pred["x_obs"],
        df_pred["x_pred"],
        "Actual x",
        "RAU x",
        "x (m)",
        "Time vs x",
    )
    # (0,2) time vs y
    ax_ty = axes[0, 2]
    _plot_with_uncertainty_band(
        ax_ty,
        df_pred["time"],
        df_pred["y_obs"],
        df_pred["y_pred"],
        "Actual y",
        "RAU y",
        "y (m)",
        "Time vs y",
    )
    # (1,0) time vs vx
    ax_vx = axes[1, 0]
    _plot_with_uncertainty_band(
        ax_vx,
        df_pred["time"],
        df_pred["vx_obs"],
        df_pred["vx_pred"],
        "Actual vx",
        "RAU vx",
        "vx (m/s)",
        "Time vs x-speed",
    )
    # (1,1) time vs vy
    ax_vy = axes[1, 1]
    _plot_with_uncertainty_band(
        ax_vy,
        df_pred["time"],
        df_pred["vy_obs"],
        df_pred["vy_pred"],
        "Actual vy",
        "RAU vy",
        "vy (m/s)",
        "Time vs y-speed",
    )
    # (1,2) GA error vs generation (running best weighted_MSE)
    ax_err = axes[1, 2]
    if error_history:
        gens = np.arange(1, len(error_history) + 1)
        ax_err.plot(gens, error_history, "k-o", markersize=4)
        # Use log scale only if values span >1 order of magnitude.
        eh = np.asarray([e for e in error_history if np.isfinite(e) and e > 0], dtype=float)
        if eh.size and (eh.max() / max(eh.min(), 1e-12) > 10.0):
            ax_err.set_yscale("log")
        ax_err.set_xlabel("GA generation")
        ax_err.set_ylabel("Best normalized loss")
        ax_err.set_title("Calibration loss vs generation")
        ax_err.grid(True, alpha=0.3, which="both")
    else:
        ax_err.text(0.5, 0.5, "No error history", ha="center", va="center", transform=ax_err.transAxes)
        ax_err.set_axis_off()
    subtitle = f"Vehicle ID {agent_id} — observed vs {comparison_model_slug} simulated"
    fig.suptitle(subtitle, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"trajectory_id_{agent_id}.png")
    plt.savefig(out_path, bbox_inches="tight")
    save_figure_to_comparison(
        fig, comparison_dir, run_id, lane_kf, agent_id, comparison_model_slug
    )
    plt.close()

##############################################################################
# 12) MAIN: load → preprocess → calibrate ALL IDs → save one CSV + plot per ID
##############################################################################

def parse_args():
    p = argparse.ArgumentParser(
        description="RAU calibration. Use --fast and --workers for quicker test runs."
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Preset: smaller GA, coarser time sampling, lower min presence (override with other flags).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel processes for GA fitness evaluation per vehicle (1=serial). Try 4–8 on multi-core CPUs.",
    )
    p.add_argument("--pop-size", type=int, default=None, help="GA population size (default: %d or 3 if --fast)" % POP_SIZE)
    p.add_argument("--generations", type=int, default=None, help="GA generations (default: %d or 4 if --fast)" % GENERATIONS)
    p.add_argument(
        "--sample-step",
        type=int,
        default=None,
        help="Use every N-th time frame (1=all frames). Larger = faster testing, fewer trajectory steps.",
    )
    p.add_argument(
        "--min-presence",
        type=int,
        default=None,
        help="Min rows per vehicle to calibrate (default: %d or 80 if --fast)" % MIN_PRESENCE,
    )
    p.add_argument(
        "--max-vehicles",
        type=int,
        default=None,
        help="Calibrate at most this many vehicles per lane (after min-presence filter); for quick tests.",
    )
    p.add_argument(
        "--vehicle-ids",
        default=None,
        metavar="IDS",
        help="Comma-separated object_ID values; if set, only these vehicles are calibrated (after min-presence filter).",
    )
    p.add_argument(
        "--ate-stop",
        type=float,
        default=None,
        help="Early-stop ATE (m) threshold; closed-loop fit (default: %.2f)" % ATE_STOP_M,
    )
    p.add_argument(
        "--no-envelope-clip",
        action="store_true",
        help="Disable hard position projection to the roadway envelope.",
    )
    p.add_argument(
        "--envelope-clip-margin",
        type=float,
        default=0.35,
        help="Same as simulation: stay this far inside L/U when clipping (m). Default: %(default)s.",
    )
    p.add_argument(
        "--no-comparison-plots",
        action="store_true",
        help="Do not mirror trajectory PNGs to comparison_visualizations/ for side-by-side model checks.",
    )
    p.add_argument(
        "--comparison-plots-dir",
        default=None,
        metavar="DIR",
        help="Folder for mirrored comparison PNGs (default: comparison_visualizations next to scripts). Pass \"\" with shell quoting to disable instead of --no-comparison-plots.",
    )
    return p.parse_args()


def main():
    try:
        _main_calibration()
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C). Terminating worker pool if any — exiting.", file=sys.stderr)
        sys.exit(130)


def _main_calibration():
    args = parse_args()
    fast = bool(args.fast)
    pop_size = args.pop_size if args.pop_size is not None else (3 if fast else POP_SIZE)
    generations = args.generations if args.generations is not None else (4 if fast else GENERATIONS)
    sample_step = args.sample_step if args.sample_step is not None else (3 if fast else SAMPLE_STEP)
    min_presence = args.min_presence if args.min_presence is not None else (80 if fast else MIN_PRESENCE)
    ate_stop = args.ate_stop if args.ate_stop is not None else ATE_STOP_M
    n_workers = max(1, int(args.workers))
    clip_to_envelope = not bool(args.no_envelope_clip)
    envelope_clip_margin_m = float(args.envelope_clip_margin)

    # Final_Lebanon_Data.csv: class 2 has median length 4.99 m (light vehicles).
    # Class 1 has median length 8.21 m (long vehicles). Do not treat code 1 as cars.
    CAR_CLASS_CODE = 2.0
    if load_envelope_boundaries():
        print("Loaded highway envelope boundaries:", "derived_boundaries/polylines_envelope.csv")
    else:
        print(
            "No envelope CSV found (run derive_boundaries_envelope.py first); "
            "RAU boundary penalty disabled until the file exists."
        )
    print("Loading and preprocessing CSV...")
    df = load_trajectory_data(CSV_PATH)
    print(f"Rows after preprocess: {len(df)}")

    df = df[df["class"] == CAR_CLASS_CODE].copy()
    print(f"Rows after car class filter (class={CAR_CLASS_CODE}): {len(df)}")

    print("Computing velocities...")
    df = compute_velocity_components(df)

    runs = sorted(df["run_id"].unique().tolist())
    direction_codes = sorted(df["lane_kf"].unique().tolist())
    print(f"Found run_id values: {runs}")
    print(f"Found direction codes (dataset column 'lane_kf'): {direction_codes}")
    print(
        f"Calibration settings: pop_size={pop_size}, generations={generations}, "
        f"sample_step={sample_step}, min_presence={min_presence}, workers={n_workers}, "
        f"fast={fast}, envelope hard clip={'ON' if clip_to_envelope else 'OFF'} "
        f"(margin {envelope_clip_margin_m} m)"
    )

    if args.no_comparison_plots:
        comparison_dir = None
    else:
        comparison_dir = resolve_comparison_output_dir(
            getattr(args, "comparison_plots_dir", None),
            _SCRIPT_DIR,
        )
    if comparison_dir:
        os.makedirs(comparison_dir, exist_ok=True)
        print(f"Trajectory comparison plots mirrored to: {comparison_dir}")

    param_names = ["Wc_cc", "eta", "xi", "tau", "v_pref", "W_boundary"]
    any_calibrated = False

    for run_id in runs:
        for direction_code in direction_codes:
            df_combo = df[
                (df["run_id"] == run_id) & (df["lane_kf"] == direction_code)
            ].copy()
            if df_combo.empty:
                continue

            # output folder for this (run_id, lane_kf)
            direction_tag = str(direction_code).replace(".", "_")
            run_tag = str(run_id).replace(".", "_")
            out_dir_combo = os.path.join(
                OUTPUT_DIR, f"run_{run_tag}_direction_{direction_tag}"
            )
            os.makedirs(out_dir_combo, exist_ok=True)

            counts = df_combo.groupby("object_ID").size()
            candidates = counts[counts >= min_presence].index.tolist()
            if getattr(args, "vehicle_ids", None):
                want = frozenset(
                    int(x.strip())
                    for x in str(args.vehicle_ids).split(",")
                    if x.strip()
                )
                candidates = [c for c in candidates if c in want]
            if args.max_vehicles is not None:
                candidates = candidates[: int(args.max_vehicles)]
            if not candidates:
                print(
                    f"Skip run_id={run_id}, direction={direction_code}: "
                    f"no IDs with >= min_presence={min_presence}."
                )
                continue

            any_calibrated = True
            print(
                f"\nCalibrating run_id={run_id}, direction={direction_code} "
                f"-> {len(candidates)} vehicles"
            )

            all_results = []
            all_predictions = []

            for k, agent_id in enumerate(candidates):
                print(f"[{k+1}/{len(candidates)}] Calibrating object_ID={agent_id} (rows={counts[agent_id]})")
                times_id = df_combo[df_combo["object_ID"] == agent_id]["time"].unique()
                df_sub = df_combo[df_combo["time"].isin(times_id)].copy()
                frame_data_sub = build_frame_data(
                    df_sub, num_nearest=NUM_NEAREST_NEIGHBORS, sample_step=sample_step
                )
                if len(frame_data_sub) < 2:
                    print("  Skip: not enough frames.")
                    continue

                best_vec, best_fit, best_mets, err_hist = run_rau_calibration(
                    frame_data_sub,
                    agent_id,
                    pop_size=pop_size,
                    generations=generations,
                    ate_stop_m=ate_stop,
                    n_workers=n_workers,
                    clip_to_envelope=clip_to_envelope,
                    envelope_clip_margin_m=envelope_clip_margin_m,
                )

                rowdict = {"id": agent_id, "best_fitness": best_fit, **best_mets}
                for i, name in enumerate(param_names):
                    rowdict[name] = best_vec[i]
                all_results.append(rowdict)

                df_pred = get_rau_predictions(
                    frame_data_sub,
                    best_vec,
                    agent_id,
                    clip_to_envelope=clip_to_envelope,
                    envelope_clip_margin_m=envelope_clip_margin_m,
                )
                if not df_pred.empty:
                    df_pred = df_pred.copy()
                    df_pred["ATE"] = best_mets["ATE"]
                    df_pred["ADE"] = best_mets["ADE"]
                    df_pred["FDE"] = best_mets["FDE"]
                    all_predictions.append(df_pred)
                    save_rau_trajectory_plot(
                        df_pred, agent_id, out_dir_combo,
                        run_id=run_id,
                        lane_kf=direction_code,
                        error_history=err_hist,
                        comparison_dir=comparison_dir,
                        comparison_model_slug="RAU",
                    )
                    print(f"  Saved plot: trajectory_id_{agent_id}.png")

            # Save combo results
            df_res = pd.DataFrame(all_results)
            out_csv = os.path.join(out_dir_combo, "rau_calibration_results.csv")
            df_res.to_csv(out_csv, index=False)
            print(f"Saved combo results: {out_csv} ({len(df_res)} rows)")

            if all_predictions:
                df_pred_all = pd.concat(all_predictions, ignore_index=True)
                out_pred_csv = os.path.join(out_dir_combo, "rau_predictions.csv")
                df_pred_all.to_csv(out_pred_csv, index=False)
                print(f"Saved combo predictions: {out_pred_csv}")

    if not any_calibrated:
        print("No calibration ran. Check MIN_PRESENCE / class filtering / run_id-lane_kf values.")

    print("Done.")


if __name__ == "__main__":
    main()
