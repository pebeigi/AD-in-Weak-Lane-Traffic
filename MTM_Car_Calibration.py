# -*- coding: utf-8 -*-
"""
Kanagaraj & Treiber (2018) — Mixed Traffic Model (MTM) calibration.

Same pipeline and outputs as PT_Car_Calibration / Delpiano_Car_Calibration: load,
GA, per-vehicle rowdict, ``mtm_car_calib_all_ids.csv``, ``mtm_car_predictions_all_ids.csv``,
trajectory plots.

The MTM is a fully 2D microscopic model designed for non-lane-disciplined / mixed traffic;
it does NOT rely on a lane centerline as an attractor. Lateral motion is driven entirely
by social forces from neighbors (MOBIL-style, with politeness for the follower contributions).

Trajectory figures are mirrored to ./comparison_visualizations/ as
run_*_lane_*_id_*_SDPM.png (same folder as PT/RL for side-by-side comparison; use
``--no-comparison-plots`` to disable).

pvec9: v0, T, s0, a, b, s0_y, s0_y_tilde, lambda, tau_lat (Tables 1-2 of the paper).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import sys
from multiprocessing import get_context

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from car_data_common import build_frame_data_with_dims
from comparison_exports import (
    resolve_comparison_output_dir,
    save_figure_to_comparison,
)
from highway_geometry import (
    clip_position_to_lane_envelope,
    envelope_lower_upper_polylines,
    load_envelope_boundaries,
)
from mtm_dynamics import mtm_acceleration_world
from PT_Car_Calibration import compute_velocities, load_and_preprocess

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_SCRIPT_DIR, "Final_Lebanon_Data.csv")

NUM_NEAREST_NEIGHBORS = 4
SAMPLE_STEP = 5
MIN_PRESENCE = 600
POP_SIZE = 10
GENERATIONS = 20
MUTATION_RATE = 0.2
ELITISM_COUNT = 4
# Closed-loop early stop: ATE in meters.
ATE_STOP_M = 2.0
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "mtm_car_calibration")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loss is normalized per-axis: SSE_pos / SS_obs_pos and SSE_speed / SS_obs_speed (unitless).
WEIGHT_POSITION = 1.0
WEIGHT_SPEED = 0.5
SIGMA_V = 0.1
TAU_OU = 0.5
SUBSTEPS = 5

PARAM_NAMES = [
    "v0",
    "T",
    "s0",
    "a",
    "b",
    "s0_y",
    "s0_y_tilde",
    "lambda",
    "tau_lat",
]

# Clamp bounds — IDM block (0-4) sized for highway speeds, MTM block (5-8) anchored
# around Kanagaraj-Treiber Table 2 values (s0_y=0.15, s̃0_y=0.30, λ=0.4 s, τ=1 s).
MTM_BOUNDS = [
    (5.0, 40.0),     # v0  [m/s]
    (0.2, 4.0),      # T   [s]
    (0.5, 6.0),      # s0  [m]
    (0.3, 4.0),      # a   [m/s²]
    (0.3, 5.0),      # b   [m/s²]
    (0.05, 1.0),     # s0_y       [m]
    (0.05, 1.5),     # s0_y_tilde [m]
    (0.05, 2.0),     # lambda     [s]
    (0.2, 5.0),      # tau_lat    [s]
]
_MUTATE_SCALES = [0.05 * (hi - lo) for (lo, hi) in MTM_BOUNDS]


def random_mtm_params():
    return [
        random.uniform(15.0, 30.0),  # v0
        random.uniform(0.5, 2.0),    # T
        random.uniform(1.5, 3.5),    # s0
        random.uniform(0.8, 2.5),    # a
        random.uniform(1.0, 3.0),    # b
        random.uniform(0.10, 0.30),  # s0_y
        random.uniform(0.20, 0.60),  # s0_y_tilde
        random.uniform(0.20, 0.80),  # lambda
        random.uniform(0.5, 2.0),    # tau_lat
    ]


def clamp_mtm_params(p):
    for i, (lo, hi) in enumerate(MTM_BOUNDS):
        p[i] = min(max(p[i], lo), hi)
    return p


def mutate_mtm_params(p, rate=0.2):
    """Per-parameter Gaussian mutation with step size = 5% of each clamp range."""
    for i in range(len(p)):
        if random.random() < rate:
            p[i] += random.gauss(0.0, _MUTATE_SCALES[i])
    return clamp_mtm_params(p)


def crossover_mtm_params(p1, p2):
    idx = random.randint(1, len(p1) - 1)
    return clamp_mtm_params(p1[:idx] + p2[idx:]), clamp_mtm_params(p2[:idx] + p1[idx:])


def simulate_mtm_single_agent(
    px,
    py,
    vx,
    vy,
    pvec9,
    neighbors,
    dest_x,
    dest_y,
    l_ego,
    w_ego,
    total_dt=1.0,
    substeps=5,
    rng: random.Random = None,
    clip_to_envelope=True,
    envelope_clip_margin_m=0.35,
    lane_kf=None,
    run_id=None,
    clip_initial=True,
):
    if rng is None:
        rng = random
    h = total_dt / substeps
    x, y, vx_, vy_ = px, py, vx, vy
    if clip_to_envelope and clip_initial and lane_kf is not None and run_id is not None:
        x, y, _ = clip_position_to_lane_envelope(
            x, y, run_id, lane_kf, margin_m=envelope_clip_margin_m
        )
    wx, wy = 0.0, 0.0
    for _ in range(substeps):
        from highway_geometry import goal_ahead

        g = goal_ahead(x, y, run_id, lane_kf)
        if g is not None:
            dest_x, dest_y = g
        ax, ay, _ = mtm_acceleration_world(
            x,
            y,
            vx_,
            vy_,
            list(neighbors),
            pvec9,
            run_id,
            lane_kf,
            dest_x,
            dest_y,
            l_ego=l_ego,
            w_ego=w_ego,
        )
        wx = wx * (1.0 - h / TAU_OU) + math.sqrt(h) * rng.gauss(0, 1)
        wy = wy * (1.0 - h / TAU_OU) + math.sqrt(h) * rng.gauss(0, 1)
        vx_ = vx_ + ax * h + SIGMA_V * wx
        vy_ = vy_ + ay * h + SIGMA_V * wy
        # Cap speeds to a generous bound; if the closed-loop sim diverges this keeps the
        # objective finite so the GA can still rank candidates rather than getting NaN-poisoned.
        if not math.isfinite(vx_):
            vx_ = 0.0
        if not math.isfinite(vy_):
            vy_ = 0.0
        # Magnitude cap at the top of the desired-speed bound, plus a margin.
        # A per-component cap of ±80 m/s lets a singular step run away at 113 m/s.
        sp = math.hypot(vx_, vy_)
        if sp > 45.0:
            vx_ *= 45.0 / sp
            vy_ *= 45.0 / sp
        # Freeway traffic in this sample does not reverse. A hard brake that
        # crosses zero otherwise accelerates the vehicle back down the corridor.
        gdx = dest_x - x
        gdy = dest_y - y
        gnorm = math.hypot(gdx, gdy)
        if gnorm > 1e-6:
            ux, uy = gdx / gnorm, gdy / gnorm
            v_long = vx_ * ux + vy_ * uy
            if v_long < 0.0:
                vx_ -= v_long * ux
                vy_ -= v_long * uy
        x += vx_ * h
        y += vy_ * h
        if not (math.isfinite(x) and math.isfinite(y)):
            x, y = px, py
            vx_, vy_ = 0.0, 0.0
            break
        if clip_to_envelope:
            x, y, _ = clip_position_to_lane_envelope(
                x, y, run_id, lane_kf, margin_m=envelope_clip_margin_m
            )
        # Logged neighbors moved with simple Euler within the substep loop (constant velocity).
        neighbors = [
            (nxw + nvx * h, nyw + nvy * h, nvx, nvy, nl, nw)
            for (nxw, nyw, nvx, nvy, nl, nw) in neighbors
        ]
    return x, y, vx_, vy_


def compute_mtm_metrics(
    frame_data,
    pvec9,
    focus_id,
    clip_to_envelope=True,
    envelope_clip_margin_m=0.35,
    rng: random.Random = None,
):
    """Whole-trip closed-loop roll-out (mirrors PT/Delpiano):
    one initial truth state, then closed-loop to end, logged neighbors at each segment;
    re-sync to truth on skipped frames (gaps).

    Per-axis squared errors on (vx, vy) — magnitude-only error would let axis-wise
    errors cancel.
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
        "Error": 1e-9,
        "ATE": 1e9,
        "ADE": 1e9,
        "FDE": 1e9,
    }
    if len(times_list) < 2:
        return bad
    if rng is None:
        from analysis.trajectory_seed import execution_rng, frame_key

        run_id, lane_kf, vid = frame_key(frame_data, focus_id)
        rng = execution_rng(run_id, lane_kf, vid, "sdpm")
    pvec9 = list(clamp_mtm_params(list(pvec9)))
    time_map = {t: {u["id"]: u for u in frame_data[t]} for t in times_list}
    t0 = times_list[0]
    if focus_id not in time_map[t0]:
        return bad
    u0 = time_map[t0][focus_id]
    x, y, vx, vy = u0["x"], u0["y"], u0["vx"], u0["vy"]
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
        id1, id2 = time_map[t1], time_map[t2]
        if focus_id not in id1 or focus_id not in id2:
            if focus_id in id2:
                ur = id2[focus_id]
                x, y, vx, vy = ur["x"], ur["y"], ur["vx"], ur["vy"]
            elif focus_id in id1:
                ur = id1[focus_id]
                x, y, vx, vy = ur["x"], ur["y"], ur["vx"], ur["vy"]
            resynced = True
            continue
        u1, u2 = id1[focus_id], id2[focus_id]
        prev_x, prev_y = x, y
        l_ego = float(u1.get("length_m", 4.5))
        w_ego = float(u1.get("width_m", 1.7))
        from_logged = (i == 0) or resynced
        resynced = False
        clip_initial = not from_logged
        pxp, pyp, vxp, vyp = simulate_mtm_single_agent(
            x,
            y,
            vx,
            vy,
            pvec9,
            list(u1["neighbors"]),
            u1["dest_x"],
            u1["dest_y"],
            l_ego,
            w_ego,
            total_dt=dt,
            substeps=SUBSTEPS,
            rng=rng,
            clip_to_envelope=clip_to_envelope,
            envelope_clip_margin_m=envelope_clip_margin_m,
            lane_kf=u1.get("lane_kf"),
            run_id=u1.get("run_id"),
            clip_initial=clip_initial,
        )
        dx_obs = u2["x"] - u1["x"]
        dy_obs = u2["y"] - u1["y"]
        step_err_dx.append((pxp - prev_x) - dx_obs)
        step_err_dy.append((pyp - prev_y) - dy_obs)
        step_obs_dx.append(dx_obs)
        step_obs_dy.append(dy_obs)
        speed_err_vx.append(vxp - u2["vx"])
        speed_err_vy.append(vyp - u2["vy"])
        speed_obs_vx.append(u2["vx"])
        speed_obs_vy.append(u2["vy"])
        disp_err_list.append(math.hypot(pxp - u2["x"], pyp - u2["y"]))
        sim_x.append(pxp)
        sim_y.append(pyp)
        obs_x.append(u2["x"])
        obs_y.append(u2["y"])
        sim_speed.append(math.hypot(vxp, vyp))
        obs_speed.append(math.hypot(u2["vx"], u2["vy"]))
        x, y, vx, vy = pxp, pyp, vxp, vyp
    if not step_err_dx:
        return bad
    step_err_dx = np.array(step_err_dx)
    step_err_dy = np.array(step_err_dy)
    step_obs_dx = np.array(step_obs_dx)
    step_obs_dy = np.array(step_obs_dy)
    n = len(step_err_dx)
    SSE_position = float(np.sum(step_err_dx**2 + step_err_dy**2))
    speed_err_vx = np.array(speed_err_vx)
    speed_err_vy = np.array(speed_err_vy)
    SSE_speed_vx = float(np.sum(speed_err_vx**2))
    SSE_speed_vy = float(np.sum(speed_err_vy**2))
    SSE_speed = SSE_speed_vx + SSE_speed_vy
    SS_obs_position = float(np.sum(step_obs_dx**2 + step_obs_dy**2)) + 1e-9
    SS_obs_speed = (
        float(np.sum(np.asarray(speed_obs_vx, dtype=float) ** 2))
        + float(np.sum(np.asarray(speed_obs_vy, dtype=float) ** 2))
        + 1e-9
    )
    if not (math.isfinite(SSE_position) and math.isfinite(SSE_speed)):
        return bad
    norm_pos = SSE_position / SS_obs_position
    norm_speed = SSE_speed / SS_obs_speed
    weighted_MSE = WEIGHT_POSITION * norm_pos + WEIGHT_SPEED * norm_speed
    if not math.isfinite(weighted_MSE):
        return bad
    SSE = SSE_position
    mean_dx = np.mean(step_obs_dx)
    mean_dy = np.mean(step_obs_dy)
    SStot = np.sum((step_obs_dx - mean_dx) ** 2 + (step_obs_dy - mean_dy) ** 2)
    r2 = 1.0 - (SSE / SStot) if SStot > 1e-9 else -999
    MSE = SSE / n
    RMSE = float(np.sqrt(MSE))
    MAE = float(np.mean(np.abs(np.concatenate([step_err_dx, step_err_dy]))))
    step_obs_norm = np.hypot(step_obs_dx, step_obs_dy)
    step_pred_dx = step_obs_dx + step_err_dx
    step_pred_dy = step_obs_dy + step_err_dy
    step_pred_norm = np.hypot(step_pred_dx, step_pred_dy)
    mask = step_obs_norm > 1e-9
    MAPE = (
        float(
            np.mean(
                np.abs((step_pred_norm[mask] - step_obs_norm[mask]) / step_obs_norm[mask])
            )
            * 100.0
        )
        if np.any(mask)
        else 1e9
    )
    dx_range = float(step_obs_dx.max() - step_obs_dx.min()) if step_obs_dx.size else 1e-9
    dy_range = float(step_obs_dy.max() - step_obs_dy.min()) if step_obs_dy.size else 1e-9
    NRMSE = RMSE / ((dx_range + dy_range) * 0.5 if (dx_range + dy_range) > 0 else 1e-9)
    disp = np.asarray(disp_err_list, dtype=float)
    return {
        "MSE": float(MSE),
        "RMSE": RMSE,
        "MAE": MAE,
        "MAPE": MAPE,
        "NRMSE": float(NRMSE),
        "SSE": float(SSE),
        "R-squared": r2,
        "Error": 1.0 / (1.0 + weighted_MSE),
        "ATE": float(np.sqrt(np.mean(disp**2))),
        "ADE": float(np.mean(disp)),
        "FDE": float(disp[-1]),
        "sim_x": sim_x,
        "sim_y": sim_y,
        "obs_x": obs_x,
        "obs_y": obs_y,
        "sim_speed": sim_speed,
        "obs_speed": obs_speed,
    }


def _rng_for_params(frame_data, focus_id, p):
    # Noise is a property of the trajectory, not of the candidate parameters.
    del p
    from analysis.trajectory_seed import execution_rng, frame_key

    run_id, lane_kf, vid = frame_key(frame_data, focus_id)
    return execution_rng(run_id, lane_kf, vid, "sdpm")


_MPF = None
_MPFID = None
_MPC = True
_MPM = 0.35


def _mp_init(fd, focus_id, clip_e, margin):
    global _MPF, _MPFID, _MPC, _MPM
    _MPF, _MPFID, _MPC, _MPM = fd, focus_id, clip_e, margin


def _mp_err(pvec):
    return compute_mtm_metrics(
        _MPF,
        pvec,
        _MPFID,
        clip_to_envelope=_MPC,
        envelope_clip_margin_m=_MPM,
        rng=_rng_for_params(_MPF, _MPFID, pvec),
    )["Error"]


def evaluate_population_par(frame_data, population, focus_id, n_workers, clip_e, margin):
    if n_workers <= 1:
        return [
            compute_mtm_metrics(
                frame_data,
                p,
                focus_id,
                clip_to_envelope=clip_e,
                envelope_clip_margin_m=margin,
                rng=_rng_for_params(frame_data, focus_id, p),
            )["Error"]
            for p in population
        ]
    ctx = get_context("spawn")
    pool = ctx.Pool(
        n_workers,
        initializer=_mp_init,
        initargs=(frame_data, focus_id, clip_e, margin),
    )
    try:
        return pool.map(_mp_err, population, chunksize=1)
    except KeyboardInterrupt:
        pool.terminate()
        raise
    finally:
        with contextlib.suppress(Exception):
            pool.close()
        with contextlib.suppress(Exception):
            pool.join()


def run_ga_mtm(
    frame_data_sub,
    focus_id,
    pop_size=POP_SIZE,
    generations=GENERATIONS,
    ate_stop_m=ATE_STOP_M,
    elitism_count=ELITISM_COUNT,
    n_workers=1,
    clip_to_envelope=True,
    envelope_clip_margin_m=0.35,
    search_seed=None,
):
    elitism = max(1, min(elitism_count, pop_size - 1))
    if search_seed is not None:
        random.seed(int(search_seed))
    else:
        from analysis.trajectory_seed import execution_seed, frame_key

        run_id, lane_kf, vid = frame_key(frame_data_sub, focus_id)
        random.seed(execution_seed(run_id, lane_kf, vid, "sdpm-search"))
    population = [clamp_mtm_params(random_mtm_params()) for _ in range(pop_size)]
    # Seed best_vec from the initial population so it is never None even if the entire first
    # generation scores at the floor (e.g., all candidates produce NaN trajectories).
    best_vec, best_score = list(population[0]), -1e9
    error_history = []
    for gen in range(generations):
        scores = evaluate_population_par(
            frame_data_sub,
            population,
            focus_id,
            n_workers,
            clip_to_envelope,
            envelope_clip_margin_m,
        )
        pop_sorted = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        if pop_sorted[0][1] > best_score:
            best_score = pop_sorted[0][1]
            best_vec = list(pop_sorted[0][0])
        err_running = (1.0 / best_score - 1.0) if best_score > 1e-12 else float("inf")
        error_history.append(float(err_running))
        cur = compute_mtm_metrics(
            frame_data_sub,
            best_vec,
            focus_id,
            clip_to_envelope=clip_to_envelope,
            envelope_clip_margin_m=envelope_clip_margin_m,
            rng=_rng_for_params(frame_data_sub, focus_id, best_vec),
        )
        if cur["ATE"] <= ate_stop_m:
            print(f"Early stop gen {gen+1}: ATE = {cur['ATE']:.2f} m")
            break
        new_pop = [pop_sorted[i][0] for i in range(elitism)]
        while len(new_pop) < pop_size:
            k = min(3, len(pop_sorted))
            a = max(random.sample(pop_sorted, k), key=lambda z: z[1])[0]
            b = max(random.sample(pop_sorted, k), key=lambda z: z[1])[0]
            c1, c2 = crossover_mtm_params(a, b)
            new_pop.append(mutate_mtm_params(c1, MUTATION_RATE))
            if len(new_pop) < pop_size:
                new_pop.append(mutate_mtm_params(c2, MUTATION_RATE))
        population = new_pop[:pop_size]
        if (gen + 1) % 10 == 0:
            print(f"GA gen={gen+1}, best fitness={best_score:.5f}")
    final = compute_mtm_metrics(
        frame_data_sub,
        best_vec,
        focus_id,
        clip_to_envelope=clip_to_envelope,
        envelope_clip_margin_m=envelope_clip_margin_m,
        rng=_rng_for_params(frame_data_sub, focus_id, best_vec),
    )
    return best_vec, best_score, final, error_history


def get_predictions_mtm(frame_data_sub, pvec, focus_id, clip_e, margin):
    times_list = sorted(frame_data_sub.keys())
    if len(times_list) < 2:
        return pd.DataFrame([])
    time_dict = {t: {u["id"]: u for u in frame_data_sub[t]} for t in times_list}
    if focus_id not in time_dict[times_list[0]]:
        return pd.DataFrame([])
    t0 = times_list[0]
    u0 = time_dict[t0][focus_id]
    x, y, vx, vy = u0["x"], u0["y"], u0["vx"], u0["vy"]
    rng = _rng_for_params(frame_data_sub, focus_id, pvec)
    p9 = list(clamp_mtm_params(list(pvec)))
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
        id1, id2 = time_dict[t1], time_dict[t2]
        if focus_id not in id1 or focus_id not in id2:
            if focus_id in id2:
                ur = id2[focus_id]
                x, y, vx, vy = ur["x"], ur["y"], ur["vx"], ur["vy"]
            elif focus_id in id1:
                ur = id1[focus_id]
                x, y, vx, vy = ur["x"], ur["y"], ur["vx"], ur["vy"]
            resynced = True
            continue
        u1, u2 = id1[focus_id], id2[focus_id]
        l_ego = float(u1.get("length_m", 4.5))
        w_ego = float(u1.get("width_m", 1.7))
        from_logged = (i == 0) or resynced
        resynced = False
        clip_initial = not from_logged
        pxp, pyp, vxp, vyp = simulate_mtm_single_agent(
            x,
            y,
            vx,
            vy,
            p9,
            list(u1["neighbors"]),
            u1["dest_x"],
            u1["dest_y"],
            l_ego,
            w_ego,
            total_dt=dt,
            substeps=SUBSTEPS,
            rng=rng,
            clip_to_envelope=clip_e,
            envelope_clip_margin_m=margin,
            lane_kf=u1.get("lane_kf"),
            run_id=u1.get("run_id"),
            clip_initial=clip_initial,
        )
        rows.append(
            {
                "id": focus_id,
                "time": t2,
                "x_obs": u2["x"],
                "y_obs": u2["y"],
                "vx_obs": u2["vx"],
                "vy_obs": u2["vy"],
                "x_pred": pxp,
                "y_pred": pyp,
                "vx_pred": vxp,
                "vy_pred": vyp,
            }
        )
        x, y, vx, vy = pxp, pyp, vxp, vyp
    return pd.DataFrame(rows)


def save_traj_plot(
    df_pred,
    agent_id,
    output_dir,
    run_id=None,
    lane_kf=None,
    error_history=None,
    comparison_dir=None,
    comparison_model_slug="SDPM",
):
    """2x3 plot: xy | t-x | t-y / t-vx | t-vy | GA error vs generation."""
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
            label="SDPM uncertainty band",
            zorder=1,
        )
        ax.plot(t, obs, "b-o", label=obs_label, markersize=3, zorder=3)
        ax.plot(t, pred, "r-+", label=pred_label, markersize=3, zorder=4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax_xy = axes[0, 0]
    polys = envelope_lower_upper_polylines(run_id, lane_kf)
    if polys is not None:
        lower, upper = polys
        ax_xy.plot(
            lower[:, 0], lower[:, 1], color="0.45", ls="-", lw=1, alpha=0.75, zorder=0,
            label="Envelope lower",
        )
        ax_xy.plot(
            upper[:, 0], upper[:, 1], color="0.45", ls="--", lw=1, alpha=0.75, zorder=0,
            label="Envelope upper",
        )
    ax_xy.plot(
        df_pred["x_obs"], df_pred["y_obs"], "b-o", label="Actual", markersize=3, zorder=2
    )
    ax_xy.plot(
        df_pred["x_pred"], df_pred["y_pred"], "r-+", label="SDPM 2D", markersize=3, zorder=2
    )
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("xy trajectory")
    ax_xy.legend(loc="best", fontsize=8)
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.3)
    for ax, c in zip((axes[0, 1], axes[0, 2]), ("x", "y")):
        _plot_with_uncertainty_band(
            ax,
            df_pred["time"],
            df_pred[f"{c}_obs"],
            df_pred[f"{c}_pred"],
            f"Actual {c}",
            f"SDPM {c}",
            f"{c} (m)",
            f"Time vs {c}",
        )
    for ax, c in zip((axes[1, 0], axes[1, 1]), ("vx", "vy")):
        _plot_with_uncertainty_band(
            ax,
            df_pred["time"],
            df_pred[f"{c}_obs"],
            df_pred[f"{c}_pred"],
            f"Actual {c}",
            f"SDPM {c}",
            f"{c} (m/s)",
            f"Time vs {c[1]}-speed",
        )
    ax_err = axes[1, 2]
    if error_history:
        gens = np.arange(1, len(error_history) + 1)
        ax_err.plot(gens, error_history, "k-o", markersize=4)
        eh = np.asarray(
            [e for e in error_history if np.isfinite(e) and e > 0], dtype=float
        )
        if eh.size and (eh.max() / max(eh.min(), 1e-12) > 10.0):
            ax_err.set_yscale("log")
        ax_err.set_xlabel("GA generation")
        ax_err.set_ylabel("Best normalized loss")
        ax_err.set_title("Calibration loss vs generation")
        ax_err.grid(True, alpha=0.3, which="both")
    else:
        ax_err.text(
            0.5, 0.5, "No error history", ha="center", va="center", transform=ax_err.transAxes
        )
        ax_err.set_axis_off()
    fig.suptitle(
        f"ID {agent_id} — Self-Driven-Particle Model", y=1.02
    )
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"trajectory_id_{agent_id}.png")
    plt.savefig(out_png, bbox_inches="tight")
    save_figure_to_comparison(
        fig, comparison_dir, run_id, lane_kf, agent_id, comparison_model_slug
    )
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Self-Driven-Particle Model 2D mixed-traffic calibration"
    )
    p.add_argument("--fast", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--pop-size", type=int, default=None)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--sample-step", type=int, default=None)
    p.add_argument("--min-presence", type=int, default=None)
    p.add_argument("--max-vehicles", type=int, default=None)
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
        help="Early-stop ATE (m) threshold (default: %.2f)" % ATE_STOP_M,
    )
    p.add_argument("--no-envelope-clip", action="store_true")
    p.add_argument("--envelope-clip-margin", type=float, default=0.35)
    p.add_argument(
        "--no-comparison-plots",
        action="store_true",
        help="Do not mirror trajectory PNGs to comparison_visualizations/ for PT vs SDPM checks.",
    )
    p.add_argument(
        "--comparison-plots-dir",
        default=None,
        metavar="DIR",
        help="Mirror comparison PNGs here (default: comparison_visualizations next to scripts).",
    )
    return p.parse_args()


def _main():
    args = parse_args()
    fast = bool(args.fast)
    pop_size = args.pop_size if args.pop_size is not None else (3 if fast else POP_SIZE)
    generations = (
        args.generations if args.generations is not None else (4 if fast else GENERATIONS)
    )
    sample_step = (
        args.sample_step if args.sample_step is not None else (3 if fast else SAMPLE_STEP)
    )
    min_presence = (
        args.min_presence if args.min_presence is not None else (80 if fast else MIN_PRESENCE)
    )
    ate_stop = args.ate_stop if args.ate_stop is not None else ATE_STOP_M
    n_workers = max(1, int(args.workers))
    clip_e = not bool(args.no_envelope_clip)
    margin = float(args.envelope_clip_margin)

    if args.no_comparison_plots:
        comparison_dir_vis = None
    else:
        comparison_dir_vis = resolve_comparison_output_dir(
            args.comparison_plots_dir, _SCRIPT_DIR
        )
    if comparison_dir_vis:
        os.makedirs(comparison_dir_vis, exist_ok=True)
        print(f"Trajectory comparison plots mirrored to: {comparison_dir_vis}")

    if load_envelope_boundaries():
        print("Loaded envelope (used for Frenet frame and hard clipping).")
    else:
        print(
            "Warning: no envelope CSV — SDPM will use destination-aligned axes (no clipping)."
        )
    print("Loading CSV...")
    df = load_and_preprocess(CSV_PATH)
    # Class 2 is the light-vehicle majority in Final_Lebanon_Data.csv (median length 4.99 m).
    # Class 1 is the long-vehicle class (median length 8.21 m).
    df = df[df["class"] == 2.0].copy()
    if "length_smoothed" not in df.columns:
        df["length_smoothed"] = 4.5
    if "width_smoothed" not in df.columns:
        df["width_smoothed"] = 1.7
    df = compute_velocities(df)
    print(
        f"MTM GA: pop={pop_size} gen={generations} step={sample_step} "
        f"workers={n_workers} clip={clip_e}"
    )
    for run_id in sorted(df["run_id"].unique().tolist()):
        for lane_val in sorted(df["lane_kf"].unique().tolist()):
            df_c = df[(df["run_id"] == run_id) & (df["lane_kf"] == lane_val)].copy()
            if df_c.empty:
                continue
            out_dir = os.path.join(
                OUTPUT_DIR, f"run_{run_id}_lane_{str(lane_val).replace('.', '_')}"
            )
            os.makedirs(out_dir, exist_ok=True)
            counts = df_c.groupby("object_ID").size()
            cands = [i for i in counts.index if counts[i] >= min_presence]
            if getattr(args, "vehicle_ids", None):
                want = frozenset(
                    int(x.strip())
                    for x in str(args.vehicle_ids).split(",")
                    if x.strip()
                )
                cands = [c for c in cands if c in want]
            if args.max_vehicles is not None:
                cands = cands[: int(args.max_vehicles)]
            if not cands:
                print(f"Skip run={run_id} lane={lane_val}")
                continue
            print(f"\nrun_id={run_id} lane={lane_val}: {len(cands)} vehicles -> {out_dir}")
            all_results = []
            all_predictions = []
            for k, agent_id in enumerate(cands):
                print(f"  [{k+1}/{len(cands)}] id={agent_id} rows={counts[agent_id]}")
                t_id = df_c[df_c["object_ID"] == agent_id]["time"].unique()
                df_s = df_c[df_c["time"].isin(t_id)].copy()
                frame_data = build_frame_data_with_dims(
                    df_s, num_nearest=NUM_NEAREST_NEIGHBORS, sample_step=sample_step
                )
                if len(frame_data) < 2:
                    print("    skip: few frames")
                    continue
                best_vec, best_fit, best_mets, err_hist = run_ga_mtm(
                    frame_data,
                    agent_id,
                    pop_size=pop_size,
                    generations=generations,
                    ate_stop_m=ate_stop,
                    n_workers=n_workers,
                    clip_to_envelope=clip_e,
                    envelope_clip_margin_m=margin,
                )
                rowdict = {"id": agent_id, "best_fitness": best_fit, **best_mets}
                for i, name in enumerate(PARAM_NAMES):
                    rowdict[name] = best_vec[i]
                all_results.append(rowdict)
                df_p = get_predictions_mtm(frame_data, best_vec, agent_id, clip_e, margin)
                if not df_p.empty:
                    df_p = df_p.copy()
                    df_p["ATE"] = best_mets["ATE"]
                    df_p["ADE"] = best_mets["ADE"]
                    df_p["FDE"] = best_mets["FDE"]
                    all_predictions.append(df_p)
                    save_traj_plot(
                        df_p,
                        agent_id,
                        out_dir,
                        run_id=run_id,
                        lane_kf=lane_val,
                        error_history=err_hist,
                        comparison_dir=comparison_dir_vis,
                        comparison_model_slug="SDPM",
                    )
                    print(f"  Saved plot: trajectory_id_{agent_id}.png")
            if all_results:
                df_res = pd.DataFrame(all_results)
                out_csv = os.path.join(out_dir, "mtm_car_calib_all_ids.csv")
                df_res.to_csv(out_csv, index=False)
                print(f"Saved combo results: {out_csv} ({len(df_res)} rows)")
            if all_predictions:
                df_pred_all = pd.concat(all_predictions, ignore_index=True)
                out_pred = os.path.join(out_dir, "mtm_car_predictions_all_ids.csv")
                df_pred_all.to_csv(out_pred, index=False)
                print(f"Saved combo predictions: {out_pred}")
    print("Done. Outputs under:", OUTPUT_DIR)


def main():
    try:
        _main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
