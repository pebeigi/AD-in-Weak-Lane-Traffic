#!/usr/bin/env python3
"""Cross-recording transfer and two cheap RAU checks.

Transfer uses the component-wise median parameter vector from one recording
and evaluates it on the held-out segment of the other recording. No re-fit.

The weight knockout sets one calibrated weight to zero and re-simulates.
It does not re-optimize the other weights.

Anticipation sensitivity re-simulates the calibrated vector at 1.6 s and 2.4 s.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

import highway_geometry as hg
from analysis.paired_pipeline import (
    RAU_NAMES,
    SDPM_NAMES,
    _load_index,
    _metrics,
    build_frames,
    _split,
)
import PT_Car_Calibration as rau

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
STAT = ROOT / "outputs" / "statistics"


def _eval(kind, frames, oid, vec):
    return _metrics(kind, frames, oid, vec)


def transfer(df, index):
    rows = []
    for train_run, test_run in ((1, 2), (2, 1)):
        src = df[df.run_id == train_run]
        dst = df[df.run_id == test_run]
        s_vec = [float(src[f"SDPM_{n}"].median()) for n in SDPM_NAMES]
        r_vec = [float(src[f"RAU_{n}"].median()) for n in RAU_NAMES]
        for _, row in dst.iterrows():
            frames = build_frames(index, int(row.run_id), int(row.lane_kf), int(row.id))
            _, test = _split(frames)
            sm = _eval("SDPM", test, int(row.id), s_vec)
            rm = _eval("RAU", test, int(row.id), r_vec)
            rows.append(
                {
                    "canonical_id": row.canonical_id,
                    "train_run": train_run,
                    "test_run": test_run,
                    "SDPM_transfer_ADE": sm["ADE"],
                    "RAU_transfer_ADE": rm["ADE"],
                    "SDPM_transfer_RMSE_s": sm["RMSE_s"],
                    "RAU_transfer_RMSE_s": rm["RMSE_s"],
                    "SDPM_transfer_RMSE_d": sm["RMSE_d"],
                    "RAU_transfer_RMSE_d": rm["RMSE_d"],
                    "SDPM_calibrated_ADE": row.SDPM_holdout_ADE,
                    "RAU_calibrated_ADE": row.RAU_holdout_ADE,
                }
            )
            print(row.canonical_id, "transfer", round(sm["ADE"], 1), round(rm["ADE"], 1), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "transferability_results.csv", index=False)
    print(
        out.groupby("test_run")[
            ["SDPM_transfer_ADE", "RAU_transfer_ADE", "SDPM_calibrated_ADE", "RAU_calibrated_ADE"]
        ]
        .median()
        .round(2)
    )
    return out


def knockout_and_horizon(df, index):
    base_T = rau.T_ANTICIPATION
    rows = []
    try:
        for _, row in df.iterrows():
            frames = build_frames(index, int(row.run_id), int(row.lane_kf), int(row.id))
            _, test = _split(frames)
            oid = int(row.id)
            full = [float(row[f"RAU_{n}"]) for n in RAU_NAMES]
            variants = {"full": full}
            no_b = list(full)
            no_b[5] = 0.0
            variants["Wb0"] = no_b
            no_c = list(full)
            no_c[0] = 0.0
            variants["Wcc0"] = no_c
            rec = {"canonical_id": row.canonical_id}
            rau.T_ANTICIPATION = base_T
            for name, vec in variants.items():
                m = _eval("RAU", test, oid, vec)
                rec[f"{name}_ADE"] = m["ADE"]
                rec[f"{name}_RMSE_d"] = m["RMSE_d"]
            for T in (1.6, 2.4):
                rau.T_ANTICIPATION = T
                m = _eval("RAU", test, oid, full)
                rec[f"Tant_{T}_ADE"] = m["ADE"]
            rau.T_ANTICIPATION = base_T
            rows.append(rec)
    finally:
        rau.T_ANTICIPATION = base_T
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rau_ablation.csv", index=False)
    print(out[["full_ADE", "Wb0_ADE", "Wcc0_ADE", "Tant_1.6_ADE", "Tant_2.4_ADE"]].median().round(3))
    return out


def main():
    hg.load_envelope_boundaries()
    index = _load_index()
    df = pd.read_csv(RES / "paired_model_results.csv")
    transfer(df, index)
    knockout_and_horizon(df, index)


if __name__ == "__main__":
    main()
