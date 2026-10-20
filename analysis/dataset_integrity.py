#!/usr/bin/env python3
"""Dataset integrity for Final_Lebanon_Data.csv.

Counts every row. Does not modify the raw file or the paired cohort.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Final_Lebanon_Data.csv"
OUT = ROOT / "outputs" / "statistics"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df["run_id"] = df["run_id"].astype(int)
    df["lane_kf"] = df["lane_kf"].astype(int)
    df["id"] = df["id"].astype(int)
    df["class"] = df["class"].astype(int)
    n_rows = len(df)
    keys = df.groupby(["run_id", "lane_kf", "id"], sort=False)
    n_traj = keys.ngroups

    class_counts = df.groupby("class").size()
    id_runs = df.groupby("id")["run_id"].nunique()
    id_lanes = df.groupby("id")["lane_kf"].nunique()
    dup_run = int((id_runs > 1).sum())
    dup_lane = int((id_lanes > 1).sum())

    # Duration and length per canonical key, plus missing-frame gaps.
    rows = []
    gap_rows = 0
    max_gap = 0.0
    for (run, lane, oid), g in keys:
        g = g.sort_values("time")
        t = g.time.to_numpy(dtype=float)
        x = g.xloc_kf.to_numpy(dtype=float)
        y = g.yloc_kf.to_numpy(dtype=float)
        dt = np.diff(t)
        # Expected step is 0.1 s. A gap is any step larger than 0.15 s.
        gaps = dt[dt > 0.15]
        gap_rows += int(gaps.size)
        if gaps.size:
            max_gap = max(max_gap, float(gaps.max()))
        path = float(np.hypot(np.diff(x), np.diff(y)).sum()) if len(g) > 1 else 0.0
        rows.append(
            {
                "run_id": int(run),
                "lane_kf": int(lane),
                "id": int(oid),
                "class": int(g["class"].mode().iloc[0]),
                "n": int(len(g)),
                "duration_s": float(t[-1] - t[0]) if len(t) else 0.0,
                "path_m": path,
                "median_speed_mps": float(g.speed_kf.median()),
                "n_gaps": int(gaps.size),
                "max_gap_s": float(gaps.max()) if gaps.size else 0.0,
            }
        )
    trips = pd.DataFrame(rows)
    trips.to_csv(OUT / "dataset_integrity_trips.csv", index=False)

    summary_rows = [
        ("raw_rows", n_rows),
        ("unique_canonical_trajectories", n_traj),
        ("ids_reused_across_run_id", dup_run),
        ("ids_reused_across_lane_kf", dup_lane),
        ("sample_interval_s", 0.1),
        ("frames_with_gap_gt_0.15s", gap_rows),
        ("max_gap_s", round(max_gap, 3)),
        ("duration_s_p50", round(float(trips.duration_s.median()), 3)),
        ("duration_s_p10", round(float(trips.duration_s.quantile(0.1)), 3)),
        ("duration_s_p90", round(float(trips.duration_s.quantile(0.9)), 3)),
        ("path_m_p50", round(float(trips.path_m.median()), 3)),
        ("path_m_p10", round(float(trips.path_m.quantile(0.1)), 3)),
        ("path_m_p90", round(float(trips.path_m.quantile(0.9)), 3)),
    ]
    for run, n in trips.groupby("run_id")["id"].nunique().items():
        summary_rows.append((f"unique_raw_id_in_run_{int(run)}", int(n)))
    for (run, lane), n in trips.groupby(["run_id", "lane_kf"]).size().items():
        summary_rows.append((f"trajectories_run{int(run)}_dir{int(lane)}", int(n)))
    for cls, n in class_counts.items():
        summary_rows.append((f"rows_class_{int(cls)}", int(n)))
    for (cls, run, lane), n in trips.groupby(["class", "run_id", "lane_kf"]).size().items():
        summary_rows.append(
            (f"trajectories_class{int(cls)}_run{int(run)}_dir{int(lane)}", int(n))
        )
    tidy = pd.DataFrame(summary_rows, columns=["statistic", "value"])
    tidy.to_csv(OUT / "dataset_integrity.csv", index=False)
    lines = [
        "Canonical trajectory key: (run_id, lane_kf, id).",
        "id is not globally unique.",
        "",
    ]
    lines += [f"{k}: {v}" for k, v in summary_rows]
    (OUT / "dataset_integrity_summary.txt").write_text("\n".join(lines) + "\n")
    print(tidy.to_string(index=False))


if __name__ == "__main__":
    main()
