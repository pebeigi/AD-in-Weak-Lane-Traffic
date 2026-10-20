#!/usr/bin/env python3
"""Holdout error growth, execution-noise replicates, and parameter-bound positions.

Stored parameters are not re-optimized. Every holdout is resimulated from the
logged state at the calibration/holdout boundary through the same metric code
used during calibration, so the default rollout reproduces the stored scores.
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paired_pipeline import RAU_NAMES, SDPM_NAMES, _load_index, _metrics, _split, build_frames
from analysis.paired_statistics import _boot_ci, _wilcoxon
from analysis.trajectory_seed import execution_seed
from MTM_Car_Calibration import MTM_BOUNDS, compute_mtm_metrics
from PT_Car_Calibration import compute_car_metrics

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
STAT = ROOT / "outputs" / "statistics"
TAB = ROOT / "outputs" / "tables"

HORIZONS = (2.0, 5.0, 10.0, 15.0)
N_REPLICATES = 20

# Clamp bounds in clamp_car_params (PT_Car_Calibration.py).
RAU_META = [
    ("Wcc", r"W_{cc}", "conflict penalty weight", "--", 10.0, 5000.0),
    ("eta", r"\eta", "value scale", "--", 0.1, 10.0),
    ("xi", r"\xi", "speed-saturation exponent", "--", 0.1, 15.0),
    ("tau", r"\tau", "execution-noise correlation time", "s", 0.1, 3.0),
    ("v_pref", r"v_{\mathrm{pref}}", "preferred speed", "m/s", 1.0, 50.0),
    ("Wb", r"W_b", "corridor penalty weight", "--", 0.0, 10000.0),
]
# Clamp bounds in MTM_BOUNDS (MTM_Car_Calibration.py), same order as SDPM_NAMES.
SDPM_META = [
    ("v0", r"v_0", "desired speed", "m/s"),
    ("T", r"T", "desired time gap", "s"),
    ("s0", r"s_0", "jam distance", "m"),
    ("a", r"a", "maximum acceleration", "m/s^2"),
    ("b", r"b", "comfortable deceleration", "m/s^2"),
    ("s0_y", r"s_0^y", "lateral attenuation scale (braking)", "m"),
    ("s0_y_tilde", r"\tilde{s}_0^y", "lateral attenuation scale (steering)", "m"),
    ("lambda", r"\lambda", "lateral sensitivity", "s"),
    ("tau_lat", r"\tau_{\mathrm{lat}}", "lateral relaxation time", "s"),
]


def _trace(kind, frames, oid, vec, rng=None):
    if kind == "RAU":
        m = compute_car_metrics(frames, vec, oid, clip_to_envelope=True, rng=rng)
    else:
        m = compute_mtm_metrics(frames, vec, oid, clip_to_envelope=True, rng=rng)
    times = sorted(frames)
    elapsed = np.asarray(times[1:]) - times[0]
    err = np.hypot(np.asarray(m["sim_x"]) - np.asarray(m["obs_x"]),
                   np.asarray(m["sim_y"]) - np.asarray(m["obs_y"]))
    return elapsed, err, float(m["ADE"])


def _job(payload):
    import highway_geometry as hg

    hg.load_envelope_boundaries()
    rec = payload["rec"]
    frames = payload["frames"]
    run_id, lane, vid = int(rec["run_id"]), int(rec["lane_kf"]), int(rec["id"])
    summary = {"canonical_id": rec["canonical_id"]}
    curves = []
    reps = []
    for kind, names in (("SDPM", SDPM_NAMES), ("RAU", RAU_NAMES)):
        vec = [float(rec[f"{kind}_{n}"]) for n in names]
        elapsed, err, ade = _trace(kind, frames, vid, vec)
        summary[f"{kind}_replayed_ADE"] = ade
        summary[f"{kind}_stored_ADE"] = float(rec[f"{kind}_holdout_ADE"])
        summary["holdout_duration_s"] = float(elapsed[-1])
        for t, e in zip(elapsed, err):
            curves.append({
                "canonical_id": rec["canonical_id"],
                "model": kind,
                "elapsed_s": float(t),
                "displacement_m": float(e),
                "holdout_duration_s": float(elapsed[-1]),
            })
        for k in range(N_REPLICATES):
            seed = execution_seed(run_id, lane, vid, f"{kind.lower()}-replicate-{k}")
            m = _metrics(kind, frames, vid, vec, rng=random.Random(seed))
            reps.append({
                "canonical_id": rec["canonical_id"],
                "model": kind,
                "replicate": k,
                **{key: m[key] for key in ("ADE", "FDE", "RMSE_s", "RMSE_d", "RMSE_speed")},
            })
    return summary, curves, reps


def _bound_row(model, name, symbol, meaning, unit, lo, hi, col):
    z = (col - lo) / (hi - lo)
    q25, q75 = np.percentile(col, [25, 75])
    zq25, zq75 = np.percentile(z, [25, 75])
    return {
        "model": model,
        "parameter": name,
        "symbol": symbol,
        "meaning": meaning,
        "units": unit,
        "lower": lo,
        "upper": hi,
        "median": float(np.median(col)),
        "iqr": float(q75 - q25),
        "z_median": float(np.median(z)),
        "z_iqr": float(zq75 - zq25),
        "frac_within_5pct_lower": float(np.mean(z <= 0.05)),
        "frac_within_5pct_upper": float(np.mean(z >= 0.95)),
        "n": int(np.isfinite(col).sum()),
    }


def parameter_bounds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, symbol, meaning, unit), (lo, hi) in zip(SDPM_META, MTM_BOUNDS):
        rows.append(_bound_row("SDPM", name, symbol, meaning, unit, lo, hi,
                               df[f"SDPM_{name}"].to_numpy(dtype=float)))
    for name, symbol, meaning, unit, lo, hi in RAU_META:
        rows.append(_bound_row("RAU", name, symbol, meaning, unit, lo, hi,
                               df[f"RAU_{name}"].to_numpy(dtype=float)))
    return pd.DataFrame(rows)


def horizon_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        at = curves[np.isclose(curves.elapsed_s, h, atol=1e-6)]
        wide = at.pivot_table(index="canonical_id", columns="model", values="displacement_m")
        wide = wide.dropna()
        if wide.empty:
            continue
        delta = (wide["RAU"] - wide["SDPM"]).to_numpy()
        lo, hi = _boot_ci(delta, np.median)
        p_w, r_rb, _ = _wilcoxon(delta)
        rows.append({
            "horizon_s": h,
            "n": int(len(wide)),
            "SDPM_median_m": float(wide["SDPM"].median()),
            "SDPM_q25": float(wide["SDPM"].quantile(0.25)),
            "SDPM_q75": float(wide["SDPM"].quantile(0.75)),
            "RAU_median_m": float(wide["RAU"].median()),
            "RAU_q25": float(wide["RAU"].quantile(0.25)),
            "RAU_q75": float(wide["RAU"].quantile(0.75)),
            "delta_median_m": float(np.median(delta)),
            "ci95_low": lo,
            "ci95_high": hi,
            "p_wilcoxon": p_w,
            "rank_biserial": r_rb,
        })
    return pd.DataFrame(rows)


def horizon_curve(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    curves = curves.assign(elapsed_s=curves.elapsed_s.round(1))
    for (model, t), sub in curves.groupby(["model", "elapsed_s"]):
        rows.append({
            "model": model,
            "elapsed_s": float(t),
            "n": int(len(sub)),
            "median_m": float(sub.displacement_m.median()),
            "q25": float(sub.displacement_m.quantile(0.25)),
            "q75": float(sub.displacement_m.quantile(0.75)),
        })
    return pd.DataFrame(rows)


def replicate_tests(reps: pd.DataFrame) -> pd.DataFrame:
    mean = reps.groupby(["canonical_id", "model"])[["ADE", "FDE", "RMSE_s", "RMSE_d", "RMSE_speed"]].mean()
    rows = []
    for key in ("ADE", "RMSE_s", "RMSE_d", "RMSE_speed", "FDE"):
        wide = mean[key].unstack("model").dropna()
        delta = (wide["RAU"] - wide["SDPM"]).to_numpy()
        lo, hi = _boot_ci(delta, np.median)
        p_w, r_rb, _ = _wilcoxon(delta)
        rows.append({
            "metric": key,
            "n": int(len(wide)),
            "replicates": N_REPLICATES,
            "sdpm_median": float(wide["SDPM"].median()),
            "rau_median": float(wide["RAU"].median()),
            "delta_median": float(np.median(delta)),
            "ci95_median_low": lo,
            "ci95_median_high": hi,
            "p_wilcoxon": p_w,
            "rank_biserial": r_rb,
            "n_rau_better": int(np.sum(delta < 0)),
            "n_sdpm_better": int(np.sum(delta > 0)),
        })
    return pd.DataFrame(rows)


def main():
    STAT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "paired_model_results.csv")
    bounds = parameter_bounds(df)
    bounds.to_csv(STAT / "parameter_bound_position.csv", index=False)
    bounds.to_csv(TAB / "parameter_bounds.csv", index=False)
    print(bounds[["model", "parameter", "median", "z_median", "frac_within_5pct_lower",
                  "frac_within_5pct_upper"]].round(3).to_string(index=False))

    index = _load_index()
    payloads = []
    for rec in df.to_dict(orient="records"):
        frames = build_frames(index, int(rec["run_id"]), int(rec["lane_kf"]), int(rec["id"]))
        _, test = _split(frames)
        payloads.append({"rec": rec, "frames": test})
    summaries, curves, reps = [], [], []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for s, c, r in pool.map(_job, payloads):
            summaries.append(s)
            curves.extend(c)
            reps.extend(r)
    summ = pd.DataFrame(summaries)
    for kind in ("SDPM", "RAU"):
        gap = (summ[f"{kind}_replayed_ADE"] - summ[f"{kind}_stored_ADE"]).abs().max()
        print(kind, "max |replayed - stored| holdout ADE:", gap)
    summ.to_csv(STAT / "replay_check.csv", index=False)

    curve_df = pd.DataFrame(curves)
    curve_df.to_csv(STAT / "horizon_curves.csv", index=False)
    hor = horizon_table(curve_df)
    hor.to_csv(STAT / "horizon_error.csv", index=False)
    hor.to_csv(TAB / "horizon_error.csv", index=False)
    horizon_curve(curve_df).to_csv(TAB / "horizon_curve_bins.csv", index=False)
    print(hor.round(3).to_string(index=False))

    rep_df = pd.DataFrame(reps)
    rep_df.to_csv(STAT / "noise_replicates.csv", index=False)
    rt = replicate_tests(rep_df)
    rt.to_csv(STAT / "noise_replicate_tests.csv", index=False)
    print(rt.round(3).to_string(index=False))
    sd = rep_df.groupby(["canonical_id", "model"]).ADE.std().unstack("model")
    print("within-trajectory SD of holdout ADE across noise replicates (median):", sd.median().round(2).to_dict())


if __name__ == "__main__":
    main()
