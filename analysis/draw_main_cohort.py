#!/usr/bin/env python3
"""Append the main split to the paired cohort.

Run once, after `paired_pipeline prepare` has drawn the pilot and reserve
splits and before any model is scored. Pilot and reserve rows are not touched.
Uses the same eligibility rule as `prepare` and draws up to 13 further
trajectories per stream with random.Random(SEED + 17) = 20261021.
"""
from __future__ import annotations

import random

import pandas as pd

from analysis.paired_pipeline import (
    OUT,
    SEED,
    _ang_diff,
    _circ_mean_deg,
    load_light,
    trip_table,
)


def main() -> None:
    cohort_path = OUT / "cohorts" / "paired_cohort.csv"
    cohort = pd.read_csv(cohort_path)
    if (cohort.split == "main").any():
        raise SystemExit("main split already present")
    df = load_light()
    trips = trip_table(df)
    parts = []
    for (run, lane), sub in trips.groupby(["run_id", "lane_kf"]):
        ang = sub.heading_deg.dropna().to_numpy()
        med = _circ_mean_deg(ang) if len(ang) else 0.0
        keep = sub[
            (sub.n >= 150)
            & (sub.path_m >= 40)
            & (sub.median_speed.between(2, 25))
            & sub.heading_deg.notna()
            & (sub.heading_deg.map(lambda h, m=med: abs(_ang_diff(h, m)) <= 25))
        ].copy()
        parts.append(keep)
    eligible = pd.concat(parts, ignore_index=True)
    rng = random.Random(SEED + 17)
    rows = []
    for (run, lane), sub in eligible.groupby(["run_id", "lane_kf"]):
        taken = set(cohort[(cohort.run_id == run) & (cohort.lane_kf == lane)].id.astype(int))
        ids = sorted(set(sub.id.astype(int)) - taken)
        n_take = min(13, len(ids))
        choose = sorted(rng.sample(ids, n_take)) if n_take else []
        meta = sub.set_index("id")
        print(f"stream {run},{lane} eligible {len(sub)} remaining {len(ids)} take {n_take}")
        for oid in choose:
            row = meta.loc[oid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            rows.append(
                {
                    "run_id": int(run),
                    "lane_kf": int(lane),
                    "id": int(oid),
                    "canonical_id": f"{int(run)}_{int(lane)}_{int(oid)}",
                    "split": "main",
                    "n": int(row.n),
                    "duration_s": float(row.duration_s),
                    "path_m": float(row.path_m),
                    "median_speed": float(row.median_speed),
                    "heading_deg": float(row.heading_deg),
                }
            )
    out = pd.concat([cohort, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(cohort_path, index=False)
    print("cohort rows", len(out), "by split", out.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
