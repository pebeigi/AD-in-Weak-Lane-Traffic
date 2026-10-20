#!/usr/bin/env python3
"""Trajectory-level interaction descriptors and their relation to model error.

Descriptors are computed from the logged tracks, not from either simulator.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paired_pipeline import OUT, _load_index, build_frames
import highway_geometry as hg

ROOT = Path(__file__).resolve().parents[1]


def _project(xy, run, lane):
    rec = hg._CORRIDORS[(int(run), int(lane))]
    cx = np.asarray(rec["x"], dtype=float)
    cy = np.asarray(rec["y"], dtype=float)
    nx = np.asarray(rec["nx"], dtype=float)
    ny = np.asarray(rec["ny"], dtype=float)
    s = np.asarray(rec["s"], dtype=float)
    a = np.column_stack([cx[:-1], cy[:-1]])
    ab = np.column_stack([np.diff(cx), np.diff(cy)])
    ab2 = np.maximum(np.sum(ab * ab, axis=1), 1e-9)
    pts = np.asarray(xy, dtype=float)
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
    ss = (1.0 - t[rows, j]) * s[j] + t[rows, j] * s[j + 1]
    return ss, lat


def descriptors_for(index, run, lane, oid) -> dict:
    frames = build_frames(index, run, lane, oid)
    times = sorted(frames)
    if len(times) < 4:
        return {}
    ego = np.array([[frames[t][0]["x"], frames[t][0]["y"]] for t in times])
    spd = np.array([math.hypot(frames[t][0]["vx"], frames[t][0]["vy"]) for t in times])
    s_e, d_e = _project(ego, run, lane)
    # nearest forward vehicle at each sample: along-track gap in (1, 40] m
    lat_off = []
    gaps = []
    n_near = []
    by_t = index[(run, lane)]
    for t, (ss, dd) in zip(times, zip(s_e, d_e)):
        block = by_t.get(float(t))
        if block is None or len(block) < 2:
            n_near.append(0)
            continue
        others = block[block[:, 0] != oid]
        if len(others) == 0:
            n_near.append(0)
            continue
        s_o, d_o = _project(others[:, 1:3], run, lane)
        gap = s_o - ss
        # same direction: gap positive means ahead along the centerline
        ahead = (gap > 1.0) & (gap <= 40.0)
        near = np.hypot(others[:, 1] - frames[t][0]["x"], others[:, 2] - frames[t][0]["y"]) <= 30.0
        n_near.append(int(near.sum()))
        if ahead.any():
            k = np.argmin(gap[ahead])
            lat_off.append(abs(float(d_o[ahead][k] - dd)))
            gaps.append(float(gap[ahead][k]))
    lat_off = np.asarray(lat_off, dtype=float)
    out = {
        "lat_pos_std_m": float(np.std(d_e, ddof=1)) if len(d_e) > 2 else float("nan"),
        "speed_std_mps": float(np.std(spd, ddof=1)) if len(spd) > 2 else float("nan"),
        "mean_neighbors_30m": float(np.mean(n_near)) if n_near else float("nan"),
        "n_forward_samples": int(len(lat_off)),
    }
    if len(lat_off):
        out["median_forward_lateral_m"] = float(np.median(lat_off))
        out["p90_forward_lateral_m"] = float(np.quantile(lat_off, 0.9))
        out["frac_forward_beyond_1_5m"] = float(np.mean(lat_off > 1.5))
        out["median_forward_gap_m"] = float(np.median(gaps))
    else:
        out["median_forward_lateral_m"] = float("nan")
        out["p90_forward_lateral_m"] = float("nan")
        out["frac_forward_beyond_1_5m"] = float("nan")
        out["median_forward_gap_m"] = float("nan")
    return out


def main() -> None:
    hg.load_envelope_boundaries()
    index = _load_index()
    cohort = pd.read_csv(OUT / "cohorts" / "paired_cohort.csv")
    rows = []
    for _, row in cohort.iterrows():
        rec = {
            "canonical_id": row.canonical_id,
            "run_id": int(row.run_id),
            "lane_kf": int(row.lane_kf),
            "split": row.split,
        }
        rec.update(descriptors_for(index, int(row.run_id), int(row.lane_kf), int(row.id)))
        rows.append(rec)
        print(row.canonical_id, rec.get("median_forward_lateral_m"), flush=True)
    desc = pd.DataFrame(rows)
    desc.to_csv(OUT / "statistics" / "interaction_descriptors.csv", index=False)
    res_path = OUT / "results" / "paired_model_results.csv"
    if not res_path.exists():
        print("descriptors written; paired results not ready")
        return
    res = pd.read_csv(res_path)
    m = desc.merge(res[["canonical_id", "delta_ADE", "delta_RMSE_d", "delta_RMSE_s"]], on="canonical_id")
    cols = [
        "median_forward_lateral_m",
        "p90_forward_lateral_m",
        "frac_forward_beyond_1_5m",
        "mean_neighbors_30m",
        "lat_pos_std_m",
        "speed_std_mps",
        "median_forward_gap_m",
    ]
    rng = np.random.default_rng(20261004)
    corr_rows = []
    for c in cols:
        for y in ("delta_ADE", "delta_RMSE_d", "delta_RMSE_s"):
            sub = m[[c, y]].dropna()
            rec = {"descriptor": c, "delta": y, "n": len(sub), "pearson_r": float("nan"),
                   "spearman_rho": float("nan"), "rho_ci95_low": float("nan"), "rho_ci95_high": float("nan")}
            if len(sub) >= 8:
                rx = sub[c].rank().to_numpy()
                ry = sub[y].rank().to_numpy()
                rec["pearson_r"] = float(sub[c].corr(sub[y]))
                rec["spearman_rho"] = float(np.corrcoef(rx, ry)[0, 1])
                idx = rng.integers(0, len(sub), size=(5000, len(sub)))
                xv, yv = sub[c].to_numpy(), sub[y].to_numpy()
                boots = []
                for row in idx:
                    bx = pd.Series(xv[row]).rank().to_numpy()
                    by = pd.Series(yv[row]).rank().to_numpy()
                    if bx.std() > 0 and by.std() > 0:
                        boots.append(np.corrcoef(bx, by)[0, 1])
                rec["rho_ci95_low"], rec["rho_ci95_high"] = (float(q) for q in np.quantile(boots, [0.025, 0.975]))
            corr_rows.append(rec)
    pd.DataFrame(corr_rows).to_csv(OUT / "statistics" / "regime_correlations.csv", index=False)
    print(pd.DataFrame(corr_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
