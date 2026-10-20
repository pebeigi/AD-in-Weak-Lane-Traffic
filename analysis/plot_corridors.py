#!/usr/bin/env python3
"""Overlay a subsample of light-vehicle tracks on the fitted corridor."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Final_Lebanon_Data.csv"
GEOM = ROOT / "outputs" / "geometry"
FIG = ROOT / "outputs" / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df = df[df["class"] == 2.0]
    rng = np.random.default_rng(20261004)
    for run, lane in ((1, 1), (1, 2), (2, 1), (2, 2)):
        sub = df[(df.run_id == run) & (df.lane_kf == lane)]
        ids = sub.id.unique()
        take = rng.choice(ids, size=min(80, len(ids)), replace=False)
        path = GEOM / f"run{run}_dir{lane}_corridor.csv"
        cor = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for oid in take:
            g = sub[sub.id == oid]
            ax.plot(g.xloc_kf, g.yloc_kf, color="0.75", lw=0.4, alpha=0.7)
        ax.plot(cor.x_lo, cor.y_lo, color="#4C78A8", lw=1.2, label="5th percentile")
        ax.plot(cor.x, cor.y, color="#333333", lw=1.4, label="Centerline")
        ax.plot(cor.x_hi, cor.y_hi, color="#F58518", lw=1.2, label="95th percentile")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"corridor_run{run}_dir{lane}.pdf")
        fig.savefig(FIG / f"corridor_run{run}_dir{lane}.png", dpi=160)
        plt.close(fig)
        print("wrote", run, lane)


if __name__ == "__main__":
    main()
