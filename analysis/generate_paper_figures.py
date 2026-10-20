#!/usr/bin/env python3
"""Publication figures for the paired SDPM–RAU comparison.

Rollouts for the case figures go through the same metric functions that
produced the stored holdout scores, so plotted paths are the scored paths.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import highway_geometry as hg
from analysis.paired_pipeline import RAU_NAMES, SDPM_NAMES, _load_index, _split, build_frames
from sdpm_calibration import compute_sdpm_metrics
from rau_calibration import compute_rau_metrics

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
RES = ROOT / "outputs" / "results"
STAT = ROOT / "outputs" / "statistics"
TAB = ROOT / "outputs" / "tables"

SDPM_C = "#3B6FA0"
RAU_C = "#E08A2E"
OBS_C = "#1A1A1A"
COL_W = 3.5
TXT_W = 7.2

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
})


def _save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _trace(kind, frames, oid, vec):
    if kind == "RAU":
        m = compute_rau_metrics(frames, vec, oid, clip_to_envelope=True)
    else:
        m = compute_sdpm_metrics(frames, vec, oid, clip_to_envelope=True)
    times = sorted(frames)
    u0 = frames[times[0]][0]
    t = np.asarray(times) - times[0]
    sx = np.r_[u0["x"], m["sim_x"]]
    sy = np.r_[u0["y"], m["sim_y"]]
    ox = np.r_[u0["x"], m["obs_x"]]
    oy = np.r_[u0["y"], m["obs_y"]]
    v0 = float(np.hypot(u0["vx"], u0["vy"]))
    sv = np.r_[v0, m["sim_speed"]]
    ov = np.r_[v0, m["obs_speed"]]
    return t, sx, sy, sv, ox, oy, ov, float(m["ADE"])


def _project(x, y, run_id, lane):
    geo = pd.read_csv(ROOT / "outputs" / "geometry" / f"run{run_id}_dir{lane}_corridor.csv")
    cx, cy, ss = geo.x.to_numpy(), geo.y.to_numpy(), geo.s.to_numpy()
    abx, aby = np.diff(cx), np.diff(cy)
    ab2 = np.maximum(abx**2 + aby**2, 1e-9)
    t = np.clip(((x[:, None] - cx[:-1]) * abx + (y[:, None] - cy[:-1]) * aby) / ab2, 0, 1)
    qx = cx[:-1] + t * abx
    qy = cy[:-1] + t * aby
    j = np.argmin((x[:, None] - qx) ** 2 + (y[:, None] - qy) ** 2, axis=1)
    r = np.arange(len(x))
    s = (1 - t[r, j]) * ss[j] + t[r, j] * ss[j + 1]
    cross = abx[j] * (y - qy[r, j]) - aby[j] * (x - qx[r, j])
    d = cross / np.sqrt(ab2[j])
    return s, d


def stagger_figure():
    ad = np.load(STAT / "nearest_forward_abs_offset.npy")
    thr = pd.read_csv(TAB / "threshold_sensitivity.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(TXT_W, 2.3))
    a.hist(ad, bins=48, range=(0, 12), color="0.35", edgecolor="white", linewidth=0.3, density=True)
    a.axvline(1.5, color="black", ls="--", lw=0.8)
    a.set_xlabel(r"Lateral offset of nearest forward vehicle, $|\Delta d|$ (m)")
    a.set_ylabel("Density")
    a.text(0.02, 0.95, "(a)", transform=a.transAxes, va="top")
    b.plot(thr.delta_m, 100 * thr.frac_outside, color="black", lw=1.2)
    for d in (1.0, 1.5, 2.0):
        y = 100 * float(thr.loc[(thr.delta_m - d).abs().idxmin(), "frac_outside"])
        b.plot(d, y, "o", color="black", ms=3)
        b.annotate(f"{y:.1f}%", (d, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    b.set_xlabel(r"Band half-width $\delta$ (m)")
    b.set_ylabel(r"Pairs with $|\Delta d|>\delta$ (%)")
    b.set_xlim(0.5, 3.0)
    b.set_ylim(0, 100)
    b.text(0.02, 0.95, "(b)", transform=b.transAxes, va="top")
    fig.tight_layout()
    _save(fig, "stagger_and_threshold")


def comparison_figure(df):
    specs = [
        ("ADE", "ADE (m)"),
        ("FDE", "FDE (m)"),
        ("RMSE_s", r"RMSE$_s$ (m)"),
        ("RMSE_d", r"RMSE$_d$ (m)"),
        ("RMSE_speed", r"Speed RMSE (m/s)"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(TXT_W, 2.0))
    for ax, (key, label) in zip(axes, specs):
        a = df[f"SDPM_holdout_{key}"].to_numpy()
        b = df[f"RAU_holdout_{key}"].to_numpy()
        bp = ax.boxplot([a, b], widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.0), whiskerprops=dict(lw=0.6),
                        capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
        for patch, color, hatch in zip(bp["boxes"], (SDPM_C, RAU_C), ("", "///")):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_hatch(hatch)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["SDPM", "RAU"])
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    fig.tight_layout(w_pad=0.6)
    _save(fig, "final_model_comparison")


def horizon_figure():
    bins = pd.read_csv(TAB / "horizon_curve_bins.csv")
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for model, color, ls in (("SDPM", SDPM_C, "-"), ("RAU", RAU_C, "--")):
        sub = bins[(bins.model == model) & (bins.n >= 20)].sort_values("elapsed_s")
        ax.fill_between(sub.elapsed_s, sub.q25, sub.q75, color=color, alpha=0.18, lw=0)
        ax.plot(sub.elapsed_s, sub.median_m, color=color, ls=ls, lw=1.3, label=model)
    n = bins[(bins.model == "SDPM") & (bins.n >= 20)].sort_values("elapsed_s")
    ax2 = ax.twinx()
    ax2.step(n.elapsed_s, n.n, where="mid", color="0.55", lw=0.7)
    ax2.set_ylabel("Trajectories (N)", color="0.45")
    ax2.tick_params(axis="y", colors="0.45")
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, 110)
    ax.set_xlabel("Elapsed holdout time (s)")
    ax.set_ylabel("Displacement error (m)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "error_by_horizon")


def delta_and_regime(df):
    desc = pd.read_csv(STAT / "interaction_descriptors.csv")
    m = df.merge(desc[["canonical_id", "frac_forward_beyond_1_5m"]], on="canonical_id")
    fig, (a, b) = plt.subplots(1, 2, figsize=(TXT_W, 2.3))
    d = df["delta_ADE"].to_numpy()
    lim = np.ceil(np.nanmax(np.abs(d)) / 10) * 10
    a.hist(d, bins=np.linspace(-lim, lim, 25), color="0.45", edgecolor="white", linewidth=0.3)
    a.axvline(0, color="black", lw=0.8)
    a.set_xlabel(r"$\Delta$ADE = ADE$_{\mathrm{RAU}}-$ADE$_{\mathrm{SDPM}}$ (m)")
    a.set_ylabel("Trajectories")
    a.text(0.02, 0.95, "(a)", transform=a.transAxes, va="top")
    b.scatter(m["frac_forward_beyond_1_5m"], m["delta_ADE"], s=9, facecolor="none", edgecolor="black", lw=0.6)
    b.axhline(0, color="0.5", lw=0.6)
    b.set_xlabel("Fraction of forward pairs with offset > 1.5 m")
    b.set_ylabel(r"$\Delta$ADE (m)")
    b.text(0.02, 0.95, "(b)", transform=b.transAxes, va="top")
    fig.tight_layout()
    _save(fig, "delta_and_regime")


def cases(df, index):
    hi = df.nlargest(1, "delta_ADE").iloc[0]
    lo = df.nsmallest(1, "delta_ADE").iloc[0]
    close = df.assign(ad=df.delta_ADE.abs()).nsmallest(1, "ad").iloc[0]
    picks = [("(a) SDPM lower ADE", hi), ("(b) RAU lower ADE", lo), ("(c) Similar ADE", close)]
    fig, axes = plt.subplots(4, 3, figsize=(TXT_W, 6.6))
    rows = []
    for col, (title, rec) in enumerate(picks):
        run, lane, oid = int(rec.run_id), int(rec.lane_kf), int(rec.id)
        frames = build_frames(index, run, lane, oid)
        _, test = _split(frames)
        sv = [float(rec[f"SDPM_{n}"]) for n in SDPM_NAMES]
        rv = [float(rec[f"RAU_{n}"]) for n in RAU_NAMES]
        t, sx, sy, ss, ox, oy, ov, s_ade = _trace("SDPM", test, oid, sv)
        _, rx, ry, rs, _, _, _, r_ade = _trace("RAU", test, oid, rv)
        rows.append({"panel": title, "canonical_id": rec.canonical_id, "SDPM_ADE": s_ade, "RAU_ADE": r_ade,
                     "stored_SDPM_ADE": rec.SDPM_holdout_ADE, "stored_RAU_ADE": rec.RAU_holdout_ADE})
        o_s, o_d = _project(ox, oy, run, lane)
        s_s, s_d = _project(sx, sy, run, lane)
        r_s, r_d = _project(rx, ry, run, lane)
        if o_s[-1] < o_s[0]:
            o_s, s_s, r_s = -o_s, -s_s, -r_s
            o_d, s_d, r_d = -o_d, -s_d, -r_d
        series = [
            ("xy", "x (m)", "y (m)"),
            ((o_s - o_s[0], s_s - o_s[0], r_s - o_s[0]), "Elapsed time (s)", r"$s-s_0$ (m)"),
            ((o_d, s_d, r_d), "Elapsed time (s)", r"$d$ (m)"),
            ((ov, ss, rs), "Elapsed time (s)", "Speed (m/s)"),
        ]
        for row, (data, xl, yl) in enumerate(series):
            ax = axes[row, col]
            if data == "xy":
                ax.plot(ox, oy, color=OBS_C, lw=1.1, label="Observed")
                ax.plot(sx, sy, color=SDPM_C, lw=1.0, label="SDPM")
                ax.plot(rx, ry, color=RAU_C, lw=1.0, ls="--", label="RAU")
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_title(f"{title}\n{rec.canonical_id}", fontsize=7.5)
            else:
                o, s, r = data
                ax.plot(t, o, color=OBS_C, lw=1.1)
                ax.plot(t, s, color=SDPM_C, lw=1.0)
                ax.plot(t, r, color=RAU_C, lw=1.0, ls="--")
            ax.set_xlabel(xl)
            if col == 0:
                ax.set_ylabel(yl)
        if col == 2:
            axes[0, col].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    _save(fig, "representative_cases")
    pd.DataFrame(rows).to_csv(TAB / "representative_cases.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "paired_model_results.csv")
    stagger_figure()
    comparison_figure(df)
    horizon_figure()
    delta_and_regime(df)
    hg.load_envelope_boundaries()
    index = _load_index()
    cases(df, index)
    print("figures written")


if __name__ == "__main__":
    main()
