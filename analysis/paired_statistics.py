#!/usr/bin/env python3
"""Paired SDPM vs RAU tests on trajectory-level holdout errors.

Delta = RAU - SDPM. Negative means RAU has the smaller error.
One row per trajectory. No frame weighting.

Framework fixed before the rerun: median paired difference with a 5000-draw
percentile bootstrap interval; Wilcoxon signed-rank with matched-pairs
rank-biserial correlation; Holm adjustment over the five metrics; mean
difference with bootstrap interval reported alongside.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
STAT = ROOT / "outputs" / "statistics"

METRICS = [
    ("ADE", "holdout displacement, m"),
    ("RMSE_s", "longitudinal RMSE, m"),
    ("RMSE_d", "lateral RMSE, m"),
    ("RMSE_speed", "speed RMSE, m/s"),
    ("FDE", "final displacement, m"),
]


def _load() -> pd.DataFrame:
    parts = []
    for name in ("pilot_paired_results.csv", "extended_paired_results.csv"):
        path = RES / name
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        raise SystemExit("no paired result files")
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("canonical_id")
    cohort = pd.read_csv(ROOT / "outputs" / "cohorts" / "paired_cohort.csv")
    if "split" not in df.columns:
        df = df.merge(cohort[["canonical_id", "split"]], on="canonical_id", how="left")
    elif df["split"].isna().any():
        df = df.drop(columns=["split"]).merge(
            cohort[["canonical_id", "split"]], on="canonical_id", how="left"
        )
    return df


def _wilcoxon(diff: np.ndarray):
    """Signed-rank test, two-sided, normal approximation with tie and continuity correction."""
    d = diff[np.isfinite(diff)]
    d = d[d != 0]
    n = len(d)
    if n < 5:
        return float("nan"), float("nan"), n
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    # tie correction
    _, counts = np.unique(ranks, return_counts=True)
    tie = float(np.sum(counts**3 - counts))
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
    sigma = math.sqrt(max(var, 1e-12))
    # continuity correction toward the mean
    z_raw = w_pos - mu
    z = (abs(z_raw) - 0.5) / sigma
    # two-sided normal tail
    p = math.erfc(z / math.sqrt(2.0))
    r_rb = (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) else float("nan")
    return p, r_rb, n


def _boot_ci(diff: np.ndarray, stat, seed=20261004, nboot=5000):
    d = diff[np.isfinite(diff)]
    if len(d) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(d, size=(nboot, len(d)), replace=True)
    stats = stat(samples, axis=1)
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def _holm(pvals):
    """Holm step-down adjusted p-values, returned in the original order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(1.0, factor * p[idx]) if math.isfinite(p[idx]) else float("nan")
        running = val if rank == 0 or not math.isfinite(running) else max(running, val)
        adj[idx] = running
    return adj


def main() -> None:
    STAT.mkdir(parents=True, exist_ok=True)
    df = _load()
    df.to_csv(RES / "paired_model_results.csv", index=False)
    rows = []
    for key, label in METRICS:
        a = df[f"SDPM_holdout_{key}"].to_numpy(dtype=float)
        b = df[f"RAU_holdout_{key}"].to_numpy(dtype=float)
        delta = b - a
        ok = np.isfinite(delta)
        d = delta[ok]
        p_w, r_rb, n = _wilcoxon(d)
        mean_lo, mean_hi = _boot_ci(d, np.mean)
        med_lo, med_hi = _boot_ci(d, np.median)
        rows.append(
            {
                "metric": key,
                "unit_note": label,
                "n": int(ok.sum()),
                "sdpm_median": float(np.nanmedian(a)),
                "rau_median": float(np.nanmedian(b)),
                "sdpm_mean": float(np.nanmean(a)),
                "rau_mean": float(np.nanmean(b)),
                "delta_mean": float(np.mean(d)),
                "delta_mean_ci_low": mean_lo,
                "delta_mean_ci_high": mean_hi,
                "delta_median": float(np.median(d)),
                "delta_sd": float(np.std(d, ddof=1)),
                "delta_iqr": float(np.subtract(*np.percentile(d, [75, 25]))),
                "ci95_median_low": med_lo,
                "ci95_median_high": med_hi,
                "p_wilcoxon": p_w,
                "rank_biserial": r_rb,
                "n_rau_better": int(np.sum(d < 0)),
                "n_sdpm_better": int(np.sum(d > 0)),
                "n_boot": 5000,
                "primary_test": "wilcoxon_signed_rank",
            }
        )
    holm = _holm([r["p_wilcoxon"] for r in rows])
    for r, p_h in zip(rows, holm):
        r["p_holm"] = float(p_h)
    out = pd.DataFrame(rows)
    out.to_csv(STAT / "paired_tests.csv", index=False)
    print(out.round(3).to_string(index=False))
    print("trajectories", len(df))


if __name__ == "__main__":
    main()
