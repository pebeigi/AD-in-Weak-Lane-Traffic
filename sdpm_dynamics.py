# -*- coding: utf-8 -*-
"""
World-frame acceleration for the adapted Kanagaraj & Treiber (2018) SDPM used
in ``sdpm_calibration``.

The published model is axis-aligned to a local road frame. Here the highway
corridor is approximated by using a **goal direction** (toward the trip
destination) as the longitudinal axis and its left normal for lateral speed,
which matches how ``sdpm_calibration`` supplies world ``(vx, vy)``
and destination points.

This is a compact realisation of the paper’s IDM-based longitudinal core with
lateral attenuation :math:`\\alpha(s^y)` (Eq.~7--8) and an OV-style lateral
relaxation (Eq.~12--16) with politeness set to zero (leaders only). Road
boundary shear (Eq.~9, 11) is omitted when no envelope geometry is loaded;
``highway_geometry`` clipping in the integrator still keeps vehicles inside a
corridor when data exist.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

# Fixed Table-2 style constant from KT2018 (not in the 9-vector GA).
LAMBDA_DELTA_W = 0.7
THETA_CONE = 0.2  # rad; |w| <= tan(theta) * |v_long| when |v| is large
IDM_DELTA = 4.0


def _goal_heading(
    x: float, y: float, dest_x: float, dest_y: float, vx: float, vy: float
) -> Tuple[float, float]:
    dx = dest_x - x
    dy = dest_y - y
    if dx * dx + dy * dy < 1e-6:
        sp = math.hypot(vx, vy)
        if sp < 1e-6:
            return 1.0, 0.0
        return vx / sp, vy / sp
    inv = 1.0 / math.hypot(dx, dy)
    return dx * inv, dy * inv


def _idm_acceleration(s: float, v: float, vl: float, v0: float, T: float, s0: float, a: float, b: float) -> float:
    """Scalar IDM acceleration along the road axis (Treiber et al. standard form)."""
    if s < 1e-3:
        return -b
    dv = v - vl
    s_star = s0 + max(0.0, v) * T + (v * dv) / (2.0 * math.sqrt(max(a * b, 1e-9)))
    a_free = a * (1.0 - (v / max(v0, 1e-6)) ** IDM_DELTA)
    a_int = -a * (s_star / s) ** 2
    return a_free + a_int


def _idm_free_acceleration(v: float, v0: float, a: float) -> float:
    return a * (1.0 - (v / max(v0, 1e-6)) ** IDM_DELTA)


def sdpm_acceleration_world(
    x: float,
    y: float,
    vx: float,
    vy: float,
    neighbors: Sequence[Tuple[float, float, float, float, float, float]],
    pvec9: Sequence[float],
    run_id,
    lane_kf,
    dest_x: float,
    dest_y: float,
    l_ego: float,
    w_ego: float,
) -> Tuple[float, float, float]:
    """
    Returns ``(ax, ay, f_long_selected)`` in m/s².

    ``neighbors`` entries: ``(nx, ny, nvx, nvy, length_j, width_j)``.
    ``pvec9``: ``[v0, T, s0, a, b, s0_y, s0_y_tilde, lambda, tau_lat]``.
    """
    del run_id, lane_kf  # reserved for envelope-based boundary terms

    v0, T, s0, a, b, s0_y, s0_y_tilde, lambda_lat, tau_lat = [float(p) for p in pvec9[:9]]
    s0_y = max(s0_y, 1e-3)
    s0_y_tilde = max(s0_y_tilde, 1e-3)
    tau_lat = max(tau_lat, 1e-3)

    ux, uy = _goal_heading(x, y, dest_x, dest_y, vx, vy)
    # Left normal (90° CCW from forward).
    px, py = -uy, ux

    v_long = vx * ux + vy * uy
    w_lat = vx * px + vy * py

    a_free = _idm_free_acceleration(v_long, v0, a)

    best_int = 0.0
    best_alpha = 0.0
    best_s = 1e9
    best_vl = v_long
    best_dy_w = 0.0
    best_wj = w_lat
    best_nj_w = w_ego
    best_a_cf_int_scalar = 0.0

    # Select dominant interaction leader (Eq. 7): argmax |alpha * a^CF,int|.
    for (nx, ny, nvx, nvy, lj, nj_w) in neighbors:
        dx = nx - x
        dy = ny - y
        r_lon = dx * ux + dy * uy
        r_lat = dx * px + dy * py
        w_bar = 0.5 * (w_ego + nj_w)
        s_y = abs(r_lat) - w_bar
        alpha_y = min(1.0, math.exp(-max(0.0, s_y) / s0_y))
        if s_y < 0.0:
            alpha_y = 1.0

        dist = math.hypot(dx, dy)
        s_bump = dist - 0.5 * (l_ego + lj) - 0.5
        s_eff = max(0.05, s_bump)

        if r_lon < -0.5 * l_ego:
            continue

        vl = nvx * ux + nvy * uy
        a_tot = _idm_acceleration(s_eff, v_long, vl, v0, T, s0, a, b)
        a_cf_int = a_tot - a_free
        # Overlapping boxes floor s at 0.05 m and make this term unbounded.
        a_cf_int = min(max(a_cf_int, -12.0), 4.0)
        score = abs(alpha_y * a_cf_int)
        if score > abs(best_alpha * best_int) or (
            abs(score - abs(best_alpha * best_int)) < 1e-9 and s_eff < best_s
        ):
            best_int = a_cf_int
            best_alpha = alpha_y
            best_s = s_eff
            best_vl = vl
            best_dy_w = r_lat
            best_wj = nvx * px + nvy * py
            best_nj_w = nj_w
            best_a_cf_int_scalar = a_cf_int

    if not neighbors:
        f_long = a_free
        g_lat = -w_lat / tau_lat
    else:
        f_long = a_free + best_alpha * best_int
        if best_s < 0.2 or (best_s < 1.0 and v_long > best_vl):
            f_long = min(f_long, -0.5 * b)

        # Lateral desired speed (Eq. 15–16, politeness p=0 leaders only).
        dy_ij = best_dy_w
        s_ij_y = abs(dy_ij) - 0.5 * (w_ego + best_nj_w)
        avg_w_half = max(0.5 * (w_ego + best_nj_w), 1e-3)

        if abs(best_a_cf_int_scalar) > 1e-6:
            # Kanagaraj and Treiber (2018), Eq. 16. With a_cf < 0 both branches
            # give a desired lateral speed away from the leader.
            if s_ij_y >= 0.0:
                atilde = math.copysign(1.0, dy_ij) * math.exp(-s_ij_y / s0_y_tilde)
            else:
                # sign(dy) * (1 + s_y / W) = dy / W, so |atilde| < 1.
                atilde = dy_ij / avg_w_half

            bracket = 1.0 - LAMBDA_DELTA_W * (best_wj - w_lat) * math.copysign(1.0, dy_ij)
            w0_ij = lambda_lat * atilde * best_a_cf_int_scalar * bracket
            # The bracket can make this desired lateral speed unstable when the
            # interaction acceleration is large. 3 m/s is already a hard lateral
            # shove relative to the speeds in this sample.
            w0 = min(max(w0_ij, -3.0), 3.0)
        else:
            w0 = 0.0

        g_lat = (w0 - w_lat) / tau_lat

    # Heading cone restriction (approximate): limit lateral jerk if too large.
    w_cap = math.tan(THETA_CONE) * max(abs(v_long), 0.5)
    if abs(w_lat) > w_cap and abs(g_lat) > 1e-6:
        g_lat *= 0.3

    # The bumper gap is floored at 0.05 m. Without a cap, an overlapping
    # neighbor makes (s*/s)^2 produce accelerations of 10^3–10^7 m/s^2 and
    # the integrator latches both velocity components at the numerical rail.
    # ±15 m/s^2 is outside the observed range of this dataset (about ±14).
    f_long = min(max(f_long, -15.0), 8.0)
    g_lat = min(max(g_lat, -8.0), 8.0)
    ax = ux * f_long + px * g_lat
    ay = uy * f_long + py * g_lat
    if not math.isfinite(ax):
        ax = 0.0
    if not math.isfinite(ay):
        ay = 0.0
    return ax, ay, f_long
