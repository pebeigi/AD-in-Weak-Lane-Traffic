#!/usr/bin/env python3
"""Reproduce the dataset table and weak-lane descriptors from Final_Lebanon_Data.csv.

Outputs are written under paper/analysis_outputs/.
No model calibration is run. Numbers in the manuscript that cite these
files must match the CSVs this script writes.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Final_Lebanon_Data.csv"
OUT = ROOT / "paper" / "analysis_outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"

# Measured from length_smoothed in this file (see tables/class_dimensions.csv).
# Calibration scripts retain class 2 (light vehicles, median length about 5 m).
CLASS_NAME = {0.0: "short", 1.0: "long", 2.0: "light"}


def _direction_axis(xy: np.ndarray) -> np.ndarray:
    X = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return vt[0]


def local_lateral(sub: pd.DataFrame, bin_m: float = 10.0) -> pd.DataFrame:
    """Arc-ordered lateral offset relative to a binned median centerline."""
    xy = sub[["xloc_kf", "yloc_kf"]].to_numpy(dtype=float)
    mu = xy.mean(axis=0)
    u = _direction_axis(xy)
    # The SVD sign is arbitrary. "Ahead" means larger s, so s must increase
    # in the direction of travel: orient u with the median net trip displacement.
    ordered = sub.sort_values("time").groupby("id")[["xloc_kf", "yloc_kf"]]
    net = (ordered.last() - ordered.first()).to_numpy(dtype=float)
    if net.size and np.median(net @ u) < 0:
        u = -u
    s = (xy - mu) @ u
    edges = np.arange(s.min(), s.max() + bin_m, bin_m)
    if len(edges) < 3:
        sub = sub.copy()
        sub["s"] = s
        sub["d_lat"] = 0.0
        return sub
    bins = np.clip(np.digitize(s, edges) - 1, 0, len(edges) - 2)
    cx = np.full(len(edges) - 1, np.nan)
    cy = np.full(len(edges) - 1, np.nan)
    for b in range(len(cx)):
        m = bins == b
        if m.sum() < 30:
            continue
        cx[b] = np.median(xy[m, 0])
        cy[b] = np.median(xy[m, 1])
    good = np.isfinite(cx)
    if good.sum() < 3:
        sub = sub.copy()
        sub["s"] = s
        sub["d_lat"] = 0.0
        return sub
    cxy = np.column_stack([cx[good], cy[good]])
    # nearest centerline vertex (bin centers are 10 m; vertex spacing is the bin)
    # Use the bin the point already fell in when that bin has a center.
    d_lat = np.full(len(xy), np.nan)
    tangents = np.zeros_like(cxy)
    idx_good = np.flatnonzero(good)
    for k, b in enumerate(idx_good):
        if k + 1 < len(idx_good):
            tangents[k] = cxy[k + 1] - cxy[k]
        elif k > 0:
            tangents[k] = cxy[k] - cxy[k - 1]
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms < 1e-6] = 1.0
    tangents = tangents / norms[:, None]
    # Keep the tangent pointing the same way along the chain.
    for k in range(1, len(tangents)):
        if tangents[k] @ tangents[k - 1] < 0:
            tangents[k] *= -1.0
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    for k in range(1, len(normals)):
        if normals[k] @ normals[k - 1] < 0:
            normals[k] *= -1.0
    # map original bin -> position in cxy
    bin_to_k = {int(b): k for k, b in enumerate(idx_good)}
    for b, k in bin_to_k.items():
        m = bins == b
        if not np.any(m):
            continue
        delta = xy[m] - cxy[k]
        d_lat[m] = delta @ normals[k]
    out = sub.copy()
    out["s"] = s
    out["d_lat"] = d_lat
    return out


def leader_offsets(frame: pd.DataFrame, horizon_m: float = 40.0) -> dict:
    """On one time slice, lateral offset of the nearest forward neighbor."""
    if len(frame) < 2 or frame["d_lat"].isna().all():
        return {"n": 0, "abs_d": np.array([]), "gap": np.array([])}
    xy = frame[["xloc_kf", "yloc_kf"]].to_numpy(dtype=float)
    s = frame["s"].to_numpy(dtype=float)
    d = frame["d_lat"].to_numpy(dtype=float)
    n = len(frame)
    abs_d = []
    gap = []
    # O(n^2) is acceptable on 1-second samples (~a few hundred vehicles).
    for i in range(n):
        if not np.isfinite(d[i]):
            continue
        ds = s - s[i]
        ahead = (ds > 1.0) & (ds < horizon_m) & np.isfinite(d)
        if not np.any(ahead):
            continue
        # nearest in Euclidean distance among vehicles ahead along s
        dist = np.hypot(xy[ahead, 0] - xy[i, 0], xy[ahead, 1] - xy[i, 1])
        j_local = int(np.argmin(dist))
        d_nb = d[np.flatnonzero(ahead)[j_local]]
        abs_d.append(abs(d_nb - d[i]))
        gap.append(float(ds[np.flatnonzero(ahead)[j_local]]))
    return {"n": len(abs_d), "abs_d": np.asarray(abs_d), "gap": np.asarray(gap)}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df["class_name"] = df["class"].map(CLASS_NAME).fillna("other")

    points = (
        df.groupby(["run_id", "lane_kf", "class_name"], sort=True)
        .size()
        .rename("points")
        .reset_index()
    )
    trips = (
        df.groupby(["run_id", "lane_kf", "class_name"], sort=True)["id"]
        .nunique()
        .rename("trips")
        .reset_index()
    )
    summary = points.merge(trips, on=["run_id", "lane_kf", "class_name"])
    summary.to_csv(TAB / "dataset_counts.csv", index=False)

    dims = (
        df.groupby("class_name")[["length_smoothed", "width_smoothed", "speed_kf"]]
        .median()
        .reset_index()
    )
    dims.to_csv(TAB / "class_dimensions.csv", index=False)

    # Kinematics at the point level, equal-point weighting (long trips weigh more).
    kin_rows = []
    for (r, lane), sub in df.groupby(["run_id", "lane_kf"]):
        kin_rows.append(
            {
                "run_id": int(r),
                "lane_kf": int(lane),
                "points": int(len(sub)),
                "trips": int(sub["id"].nunique()),
                "speed_p50": float(sub.speed_kf.median()),
                "speed_p90": float(sub.speed_kf.quantile(0.9)),
                "abs_acc_p50": float(sub.acceleration_kf.abs().median()),
                "abs_acc_p95": float(sub.acceleration_kf.abs().quantile(0.95)),
            }
        )
    pd.DataFrame(kin_rows).to_csv(TAB / "kinematics_by_stream.csv", index=False)

    # Local lateral coordinate and leader offset. Subsample times to 1 s.
    lat_rows = []
    leader_abs = []
    leader_gap = []
    hist_collect = []
    for (r, lane), sub in df.groupby(["run_id", "lane_kf"]):
        located = local_lateral(sub)
        d = located["d_lat"].to_numpy()
        finite = d[np.isfinite(d)]
        if finite.size == 0:
            continue
        lat_rows.append(
            {
                "run_id": int(r),
                "lane_kf": int(lane),
                "d_p02": float(np.quantile(finite, 0.02)),
                "d_p50": float(np.quantile(finite, 0.50)),
                "d_p98": float(np.quantile(finite, 0.98)),
                "width_p02_p98_m": float(np.quantile(finite, 0.98) - np.quantile(finite, 0.02)),
                "d_std_m": float(np.std(finite)),
            }
        )
        hist_collect.append(finite)
        # 1-second slices
        times = np.sort(located["time"].unique())
        # times are 0.1 s; take every 10th unique time present
        take = times[::10]
        abs_d_all = []
        gap_all = []
        for t in take:
            fr = located[located["time"] == t]
            res = leader_offsets(fr)
            if res["n"]:
                abs_d_all.append(res["abs_d"])
                gap_all.append(res["gap"])
        if abs_d_all:
            ad = np.concatenate(abs_d_all)
            gp = np.concatenate(gap_all)
            leader_abs.append(ad)
            leader_gap.append(gp)
            lat_rows[-1]["leader_n"] = int(ad.size)
            lat_rows[-1]["leader_abs_d_p50"] = float(np.median(ad))
            lat_rows[-1]["leader_abs_d_p90"] = float(np.quantile(ad, 0.9))
            lat_rows[-1]["frac_leader_abs_d_gt_1p5m"] = float(np.mean(ad > 1.5))
            lat_rows[-1]["gap_p50"] = float(np.median(gp))

    lat_df = pd.DataFrame(lat_rows)
    lat_df.to_csv(TAB / "lateral_and_leaders.csv", index=False)

    # Figure: lateral histograms
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    keys = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for ax, (r, lane) in zip(axes.ravel(), keys):
        sub = df[(df.run_id == r) & (df.lane_kf == lane)]
        located = local_lateral(sub)
        d = located["d_lat"].to_numpy()
        d = d[np.isfinite(d)]
        ax.hist(d, bins=60, color="0.25", density=True)
        ax.set_title(f"run {r}, direction code {lane}")
        ax.set_xlabel("Local lateral offset (m)")
        ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(FIG / "lateral_offset_hist.pdf")
    fig.savefig(FIG / "lateral_offset_hist.png", dpi=200)
    plt.close(fig)

    if leader_abs:
        ad = np.concatenate(leader_abs)
        stat_dir = ROOT / "outputs" / "statistics"
        stat_dir.mkdir(parents=True, exist_ok=True)
        np.save(stat_dir / "nearest_forward_abs_offset.npy", ad)
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.hist(ad, bins=40, range=(0, 12), color="0.25", density=True)
        ax.axvline(1.5, color="0.45", ls="--", lw=1)
        ax.set_xlabel("Lateral offset of nearest forward neighbor (m)")
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(FIG / "leader_lateral_offset.pdf")
        fig.savefig(FIG / "leader_lateral_offset.png", dpi=200)
        plt.close(fig)
        pd.DataFrame(
            {
                "leader_abs_d_p50": [float(np.median(ad))],
                "leader_abs_d_p90": [float(np.quantile(ad, 0.9))],
                "frac_gt_1p5m": [float(np.mean(ad > 1.5))],
                "n_pairs": [int(ad.size)],
                "gap_p50": [float(np.median(np.concatenate(leader_gap)))],
            }
        ).to_csv(TAB / "leader_offset_pooled.csv", index=False)

    print("wrote", TAB)
    print(summary.to_string(index=False))
    print(lat_df.to_string(index=False))


if __name__ == "__main__":
    main()
