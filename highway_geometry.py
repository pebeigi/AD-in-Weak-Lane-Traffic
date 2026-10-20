# -*- coding: utf-8 -*-
"""Corridor geometry loaded from outputs/geometry/run{R}_dir{D}_corridor.csv.

Until those files exist, clipping and boundary risk stay off.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_GEOM = _ROOT / "outputs" / "geometry"
_CORRIDORS: dict[tuple[int, int], dict] = {}


def _key(run_id, lane_kf):
    try:
        return int(float(run_id)), int(float(lane_kf))
    except (TypeError, ValueError):
        return None


def load_envelope_boundaries() -> bool:
    _CORRIDORS.clear()
    if not _GEOM.is_dir():
        return False
    found = False
    for path in sorted(_GEOM.glob("run*_dir*_corridor.csv")):
        name = path.stem  # run1_dir2_corridor
        parts = name.split("_")
        if len(parts) < 3:
            continue
        run = int(parts[0].replace("run", ""))
        direction = int(parts[1].replace("dir", ""))
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        _CORRIDORS[(run, direction)] = {
            "s": arr[:, 0],
            "x": arr[:, 1],
            "y": arr[:, 2],
            "nx": arr[:, 3],
            "ny": arr[:, 4],
            "x_lo": arr[:, 5],
            "y_lo": arr[:, 6],
            "x_hi": arr[:, 7],
            "y_hi": arr[:, 8],
        }
        found = True
    return found


def _poly(key, which):
    rec = _CORRIDORS.get(key)
    if rec is None:
        return None
    if which == "lo":
        return np.column_stack([rec["x_lo"], rec["y_lo"]])
    if which == "hi":
        return np.column_stack([rec["x_hi"], rec["y_hi"]])
    return np.column_stack([rec["x"], rec["y"]])


def lane_boundary_polylines_m(run_id, lane_kf):
    key = _key(run_id, lane_kf)
    if key is None or key not in _CORRIDORS:
        return None
    return _poly(key, "lo"), _poly(key, "hi")


def envelope_lower_upper_polylines(run_id, lane_kf):
    return lane_boundary_polylines_m(run_id, lane_kf)


def inside_clearance(px, py, run_id, lane_kf):
    """Signed distance to the nearer percentile boundary.

    Positive inside the corridor, negative outside. Distance to the polyline
    itself is not used: that distance is small on both sides of the line.
    """
    key = _key(run_id, lane_kf)
    rec = _CORRIDORS.get(key) if key is not None else None
    if rec is None or len(rec["x"]) < 2:
        return None
    cx = np.asarray(rec["x"], dtype=float)
    cy = np.asarray(rec["y"], dtype=float)
    nx = np.asarray(rec["nx"], dtype=float)
    ny = np.asarray(rec["ny"], dtype=float)
    lo = (rec["x_lo"] - cx) * nx + (rec["y_lo"] - cy) * ny
    hi = (rec["x_hi"] - cx) * nx + (rec["y_hi"] - cy) * ny
    a = np.column_stack([cx[:-1], cy[:-1]])
    ab = np.column_stack([np.diff(cx), np.diff(cy)])
    ab2 = np.maximum(np.sum(ab * ab, axis=1), 1e-9)
    px_b, py_b = np.broadcast_arrays(np.asarray(px, dtype=float), np.asarray(py, dtype=float))
    shape = px_b.shape
    pts = np.column_stack([px_b.ravel(), py_b.ravel()])
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / ab2[None, :], 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    diff = pts[:, None, :] - proj
    j = np.argmin(np.sum(diff * diff, axis=2), axis=1)
    rows = np.arange(len(pts))
    chosen_ab = ab[j]
    chosen = diff[rows, j]
    cross = chosen_ab[:, 0] * chosen[:, 1] - chosen_ab[:, 1] * chosen[:, 0]
    lat = cross / np.sqrt(ab2[j])
    tt = t[rows, j]
    lo_q = (1.0 - tt) * lo[j] + tt * lo[j + 1]
    hi_q = (1.0 - tt) * hi[j] + tt * hi[j + 1]
    clear = np.minimum(lat - lo_q, hi_q - lat)
    return clear.reshape(shape)


def boundary_prob_array(ex, ey, run_id, lane_kf, threshold_m, sigma_b):
    clear = inside_clearance(ex, ey, run_id, lane_kf)
    if clear is None:
        return np.zeros_like(np.asarray(ex, dtype=float))
    sigma = np.maximum(np.asarray(sigma_b, dtype=float), 1e-6)
    z = (clear - float(threshold_m)) / sigma
    from math import erf

    erf_v = np.vectorize(erf, otypes=[float])
    return 0.5 * (1.0 - np.asarray(erf_v(z / math.sqrt(2.0)), dtype=float))


def road_frame_rmse(sim_x, sim_y, obs_x, obs_y, sim_speed, obs_speed, run_id, lane_kf):
    """Decompose closed-loop position error in the local centerline frame.

    Longitudinal and lateral components use the centerline tangent and the
    stored normal at the vertex nearest the observed point. Speed RMSE is the
    root-mean-square difference of speed magnitudes, in m/s.
    """
    sx = np.asarray(sim_x, dtype=float)
    sy = np.asarray(sim_y, dtype=float)
    ox = np.asarray(obs_x, dtype=float)
    oy = np.asarray(obs_y, dtype=float)
    sv = np.asarray(sim_speed, dtype=float)
    ov = np.asarray(obs_speed, dtype=float)
    nan3 = (float("nan"), float("nan"), float("nan"))
    if sx.size == 0 or not np.isfinite(sx).all():
        return nan3
    key = _key(run_id, lane_kf)
    rec = _CORRIDORS.get(key) if key is not None else None
    ex = sx - ox
    ey = sy - oy
    if rec is None or len(rec["x"]) < 2:
        return nan3
    cx = rec["x"]
    cy = rec["y"]
    d2 = (ox[:, None] - cx[None, :]) ** 2 + (oy[:, None] - cy[None, :]) ** 2
    i = np.argmin(d2, axis=1)
    nx = rec["nx"][i]
    ny = rec["ny"][i]
    # Stored normal is (-ty, tx) for a tangent (tx, ty) along increasing s.
    tx = ny
    ty = -nx
    e_s = ex * tx + ey * ty
    e_d = ex * nx + ey * ny
    rmse_s = float(np.sqrt(np.mean(e_s**2)))
    rmse_d = float(np.sqrt(np.mean(e_d**2)))
    rmse_v = float(np.sqrt(np.mean((sv - ov) ** 2)))
    return rmse_s, rmse_d, rmse_v


def goal_ahead(x, y, run_id, lane_kf, ahead_m=40.0):
    """Point ahead_m along the centerline from the nearest centerline vertex."""
    key = _key(run_id, lane_kf)
    poly = _poly(key, "center") if key is not None else None
    if poly is None or len(poly) < 2:
        return None
    d = np.hypot(poly[:, 0] - x, poly[:, 1] - y)
    i = int(np.argmin(d))
    step = np.hypot(np.diff(poly[:, 0]), np.diff(poly[:, 1]))
    acc = 0.0
    j = i
    while j < len(poly) - 1 and acc < ahead_m:
        acc += float(step[j])
        j += 1
    if j == i and i > 0:
        j = min(len(poly) - 1, i + 1)
    return float(poly[j, 0]), float(poly[j, 1])


def clip_position_to_lane_envelope(x, y, run_id, lane_kf, margin_m=0.35):
    """Keep a point inside the percentile corridor, inset by margin_m.

    The inset is applied along the local centerline normal. Distance to the
    nearest centerline vertex is not a corridor coordinate: on a 15 m vertex
    spacing that distance is several meters even for a point in the roadway.
    """
    key = _key(run_id, lane_kf)
    rec = _CORRIDORS.get(key) if key is not None else None
    if rec is None or len(rec["x"]) < 2:
        return float(x), float(y), False
    cx = np.asarray(rec["x"], dtype=float)
    cy = np.asarray(rec["y"], dtype=float)
    nx = np.asarray(rec["nx"], dtype=float)
    ny = np.asarray(rec["ny"], dtype=float)
    lo = (rec["x_lo"] - cx) * nx + (rec["y_lo"] - cy) * ny
    hi = (rec["x_hi"] - cx) * nx + (rec["y_hi"] - cy) * ny
    a = np.column_stack([cx[:-1], cy[:-1]])
    ab = np.column_stack([np.diff(cx), np.diff(cy)])
    ab2 = np.maximum(np.sum(ab * ab, axis=1), 1e-9)
    apx = float(x) - a[:, 0]
    apy = float(y) - a[:, 1]
    t = np.clip((apx * ab[:, 0] + apy * ab[:, 1]) / ab2, 0.0, 1.0)
    projx = a[:, 0] + t * ab[:, 0]
    projy = a[:, 1] + t * ab[:, 1]
    dx = float(x) - projx
    dy = float(y) - projy
    j = int(np.argmin(dx * dx + dy * dy))
    abj = ab[j]
    cross = abj[0] * dy[j] - abj[1] * dx[j]
    lat = cross / math.sqrt(float(ab2[j]))
    tt = float(t[j])
    lo_q = (1.0 - tt) * lo[j] + tt * lo[j + 1]
    hi_q = (1.0 - tt) * hi[j] + tt * hi[j + 1]
    margin = float(margin_m)
    lo_m = lo_q + margin
    hi_m = hi_q - margin
    if lo_m > hi_m:
        target = 0.5 * (lo_q + hi_q)
    else:
        target = min(max(lat, lo_m), hi_m)
    if abs(target - lat) < 1e-6:
        return float(x), float(y), False
    inv = 1.0 / math.sqrt(float(ab2[j]))
    # Left normal of the segment, the same sign as the signed offset.
    nnx = -abj[1] * inv
    nny = abj[0] * inv
    return float(projx[j] + target * nnx), float(projy[j] + target * nny), True
