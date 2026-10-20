#!/usr/bin/env python3
"""Fraction of nearest-forward pairs outside a lateral band of half-width delta.

Uses the same centerline and pair definition as analysis/dataset_and_behavior.py,
with the corrected stream-axis orientation (n = 239,233 pairs). All vehicle classes are included.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.dataset_and_behavior import DATA, leader_offsets, local_lateral

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
STAT = ROOT / "outputs" / "statistics"


def collect() -> np.ndarray:
    cache = STAT / "nearest_forward_abs_offset.npy"
    if cache.exists():
        return np.load(cache)
    df = pd.read_csv(DATA)
    chunks = []
    for (r, lane), sub in df.groupby(["run_id", "lane_kf"]):
        print("stream", int(r), int(lane), "rows", len(sub), flush=True)
        located = local_lateral(sub)
        times = np.sort(located["time"].unique())[::10]
        for t in times:
            res = leader_offsets(located[located["time"] == t])
            if res["n"]:
                chunks.append(res["abs_d"])
    ad = np.concatenate(chunks) if chunks else np.array([])
    STAT.mkdir(parents=True, exist_ok=True)
    np.save(cache, ad)
    return ad


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    ad = collect()
    print("n", ad.size, "median", float(np.median(ad)))
    deltas = np.round(np.arange(0.5, 3.01, 0.1), 2)
    rows = []
    for delta in deltas:
        rows.append({
            "delta_m": float(delta),
            "n": int(ad.size),
            "frac_outside": float(np.mean(ad > delta)),
        })
    table = pd.DataFrame(rows)
    table.to_csv(TAB / "threshold_sensitivity.csv", index=False)
    table.to_csv(STAT / "threshold_sensitivity.csv", index=False)
    for d in (1.0, 1.5, 2.0):
        hit = table.loc[(table.delta_m - d).abs().idxmin()]
        print(f"delta {hit.delta_m:.1f} outside {hit.frac_outside:.4f}")

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(table.delta_m, 100 * table.frac_outside, color="#333333", lw=1.4)
    for d in (1.0, 1.5, 2.0):
        y = 100 * float(table.loc[(table.delta_m - d).abs().idxmin(), "frac_outside"])
        ax.plot(d, y, "o", color="#333333", ms=3.5)
    ax.set_xlabel("Lateral-band half-width (m)")
    ax.set_ylabel("Pairs outside the band (%)")
    ax.set_xlim(0.5, 3.0)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIG / "threshold_sensitivity.pdf")
    fig.savefig(FIG / "threshold_sensitivity.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
