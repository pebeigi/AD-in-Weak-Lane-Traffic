#!/usr/bin/env python3
"""Dataset characterization figures, table rows, and macros.

Reads Final_Lebanon_Data.csv (not modified) and the stream corridors in
outputs/geometry. Writes:
  outputs/statistics/dataset_by_stream.csv
  outputs/figures/dataset_tracks.{pdf,png}
  outputs/figures/dataset_timespace.{pdf,png}
  outputs/figures/dataset_distributions.{pdf,png}
  outputs/figures/dataset_snapshot.{pdf,png}
  paper/tables/tab_dataset.tex
  paper/dataset_numbers.tex
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Final_Lebanon_Data.csv"
GEOM = ROOT / "outputs" / "geometry"
FIG = ROOT / "outputs" / "figures"
STAT = ROOT / "outputs" / "statistics"
PAPER = ROOT / "paper"
SEED = 20261004
STREAMS = ((1, 1), (1, 2), (2, 1), (2, 2))
DIR_COLOR = {1: "#4C78A8", 2: "#F58518"}
CLASS_NAME = {0: "short", 1: "long", 2: "light"}
S_LO, S_HI = 100.0, 350.0

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    for c in ("run_id", "lane_kf", "id", "class"):
        df[c] = df[c].astype(int)
    return df.sort_values(["run_id", "lane_kf", "id", "time"]).reset_index(drop=True)


def corridor(run: int, lane: int) -> pd.DataFrame:
    return pd.read_csv(GEOM / f"run{run}_dir{lane}_corridor.csv")


def project(x: np.ndarray, y: np.ndarray, cor: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Along-road s and signed lateral d (left of travel) on the stream centerline."""
    cx, cy = cor.x.to_numpy(), cor.y.to_numpy()
    cs = cor.s.to_numpy()
    tx, ty = np.gradient(cx), np.gradient(cy)
    nrm = np.hypot(tx, ty)
    tx, ty = tx / nrm, ty / nrm
    s = np.empty_like(x)
    d = np.empty_like(x)
    step = 200_000
    for a in range(0, len(x), step):
        px, py = x[a:a + step, None], y[a:a + step, None]
        k = np.argmin((px - cx) ** 2 + (py - cy) ** 2, axis=1)
        dx, dy = x[a:a + step] - cx[k], y[a:a + step] - cy[k]
        s[a:a + step] = cs[k] + dx * tx[k] + dy * ty[k]
        d[a:a + step] = -dx * ty[k] + dy * tx[k]
    return s, d


def trajectory_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["run_id", "lane_kf", "id"], sort=False)
    out = g.agg(cls=("class", lambda c: int(c.mode().iloc[0])), n=("time", "size"),
                t0=("time", "min"), t1=("time", "max"),
                length=("length_smoothed", "median"), width=("width_smoothed", "median"),
                v50=("speed_kf", "median")).reset_index()
    out["duration"] = out.t1 - out.t0
    return out


def stream_summary(df: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, lane in STREAMS:
        sub = df[(df.run_id == run) & (df.lane_kf == lane)]
        tr = trips[(trips.run_id == run) & (trips.lane_kf == lane)]
        conc = sub.groupby("time").size()
        rows.append({
            "run_id": run, "lane_kf": lane, "rows": len(sub), "trajectories": len(tr),
            "light": int((tr.cls == 2).sum()), "long": int((tr.cls == 1).sum()),
            "short": int((tr.cls == 0).sum()),
            "t_first_s": float(sub.time.min()), "t_last_s": float(sub.time.max()),
            "dur_p50_s": float(tr.duration.median()),
            "share_under_5s": float((tr.duration < 5).mean() * 100),
            "speed_p50": float(sub.speed_kf.median()), "speed_p90": float(sub.speed_kf.quantile(0.9)),
            "conc_p50": float(conc.median()), "conc_max": int(conc.max()),
            "heading_deg": float(np.degrees(np.arctan2(
                np.median(sub.groupby("id").yloc_kf.agg(lambda v: v.iloc[-1] - v.iloc[0])),
                np.median(sub.groupby("id").xloc_kf.agg(lambda v: v.iloc[-1] - v.iloc[0]))))),
        })
    return pd.DataFrame(rows)


def fig_tracks(df: pd.DataFrame, trips: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), gridspec_kw={"height_ratios": [1.25, 1]})
    for k, run in enumerate((1, 2)):
        ax = axes[0, k]
        for lane in (1, 2):
            tr = trips[(trips.run_id == run) & (trips.lane_kf == lane) & (trips.duration >= 5)]
            take = set(rng.choice(tr.id.to_numpy(), size=min(250, len(tr)), replace=False))
            sub = df[(df.run_id == run) & (df.lane_kf == lane) & (df.id.isin(take))]
            for _, g in sub.groupby("id", sort=False):
                ax.plot(g.xloc_kf, g.yloc_kf, color=DIR_COLOR[lane], lw=0.3, alpha=0.45)
            ax.plot([], [], color=DIR_COLOR[lane], lw=1.2, label=f"Direction {lane}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"({'ab'[k]}) Recording {run}: 250 trajectories per direction")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(frameon=False, loc="upper left")
    # Speed profiles of seeded random long light-vehicle trajectories, recording 1.
    for k, lane in enumerate((1, 2)):
        ax = axes[1, k]
        tr = trips[(trips.run_id == 1) & (trips.lane_kf == lane) & (trips.cls == 2) & (trips.duration >= 60)]
        take = rng.choice(tr.id.to_numpy(), size=5, replace=False)
        for j, oid in enumerate(take):
            g = df[(df.run_id == 1) & (df.lane_kf == lane) & (df.id == oid)]
            ax.plot(g.time - g.time.iloc[0], g.speed_kf, lw=0.9, color=plt.cm.viridis(j / 5))
        ax.set_title(f"({'cd'[k]}) Recording 1, direction {lane}: speed of five light vehicles")
        ax.set_xlabel("Time since first observation (s)")
        ax.set_ylabel(r"Speed (m s$^{-1}$)")
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG / "dataset_tracks.pdf")
    fig.savefig(FIG / "dataset_tracks.png", dpi=170)
    plt.close(fig)


def fig_timespace(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=False)
    im = None
    for ax, (run, lane) in zip(axes.ravel(), STREAMS):
        sub = df[(df.run_id == run) & (df.lane_kf == lane)]
        cor = corridor(run, lane)
        s, d = project(sub.xloc_kf.to_numpy(float), sub.yloc_kf.to_numpy(float), cor)
        keep = (np.abs(d) <= 15) & (s >= 0) & (s <= cor.s.max())
        t = sub.time.to_numpy(float)[keep]
        v = sub.speed_kf.to_numpy(float)[keep]
        s = s[keep]
        tb = np.arange(0, t.max() + 10, 10.0)
        sb = np.arange(0, s.max() + 10, 10.0)
        tot, _, _ = np.histogram2d(s, t, bins=[sb, tb], weights=v)
        cnt, _, _ = np.histogram2d(s, t, bins=[sb, tb])
        with np.errstate(invalid="ignore"):
            mean = np.where(cnt >= 5, tot / np.maximum(cnt, 1), np.nan)
        im = ax.pcolormesh(tb, sb, mean, cmap="RdYlGn", vmin=0, vmax=20, shading="flat", rasterized=True)
        ax.set_title(f"Recording {run}, direction {lane}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Along-road position $s$ (m)")
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cax = fig.add_axes((0.915, 0.12, 0.018, 0.78))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"Mean speed (m s$^{-1}$)")
    fig.savefig(FIG / "dataset_timespace.pdf", dpi=200)
    fig.savefig(FIG / "dataset_timespace.png", dpi=170)
    plt.close(fig)


def fig_distributions(df: pd.DataFrame, trips: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axes[0]
    bins = np.arange(0, 30.5, 0.5)
    for run, lane in STREAMS:
        v = df[(df.run_id == run) & (df.lane_kf == lane)].speed_kf
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.0,
                color=DIR_COLOR[lane], ls="-" if run == 1 else "--", label=f"R{run} D{lane}")
    ax.set_xlabel(r"Speed (m s$^{-1}$)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Speed, all rows")
    ax.legend(frameon=False)
    ax = axes[1]
    for run in (1, 2):
        dur = np.sort(trips[trips.run_id == run].duration.to_numpy())
        ax.plot(dur, np.arange(1, len(dur) + 1) / len(dur), color="0.2", ls="-" if run == 1 else "--",
                label=f"Recording {run}")
    for xv in (5, 15):
        ax.axvline(xv, color="0.6", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Trajectory duration (s)")
    ax.set_ylabel("Cumulative share")
    ax.set_title("(b) Trajectory duration")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[2]
    marker = {0: "^", 1: "s", 2: "."}
    for run, col in ((1, "#4C78A8"), (2, "#E45756")):
        for cls in (2, 1, 0):
            t = trips[(trips.run_id == run) & (trips.cls == cls)]
            if len(t) > 600:
                t = t.sample(600, random_state=SEED)
            ax.scatter(t.length, t.width, s=3 if cls == 2 else 6, marker=marker[cls], color=col,
                       alpha=0.5, lw=0)
    for run, col in ((1, "#4C78A8"), (2, "#E45756")):
        ax.scatter([], [], color=col, s=10, label=f"Recording {run}")
    ax.set_xlabel("Median length (m)")
    ax.set_ylabel("Median width (m)")
    ax.set_title("(c) Stored dimensions by trajectory")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "dataset_distributions.pdf", dpi=200)
    fig.savefig(FIG / "dataset_distributions.png", dpi=170)
    plt.close(fig)


def fig_snapshot(df: pd.DataFrame) -> dict:
    """Plan view of one time step in the corridor frame, recording 1, both directions."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.4))
    info = {}
    for ax, lane in zip(axes, (1, 2)):
        sub = df[(df.run_id == 1) & (df.lane_kf == lane)]
        cor = corridor(1, lane)
        s, d = project(sub.xloc_kf.to_numpy(float), sub.yloc_kf.to_numpy(float), cor)
        sub = sub.assign(s=s, d=d)
        win = sub[(sub.s >= S_LO) & (sub.s <= S_HI) & (sub.d.abs() <= 8)]
        counts = win.groupby("time").size()
        cand = counts[counts >= counts.quantile(0.9)].index.to_numpy()
        t0 = float(rng.choice(cand))
        snap = win[np.isclose(win.time, t0)]
        for _, r in snap.iterrows():
            col = "#4C78A8" if r["class"] == 2 else ("#54A24B" if r["class"] == 0 else "#B279A2")
            ax.add_patch(Rectangle((r.s - r.length_smoothed / 2, r.d - r.width_smoothed / 2),
                                   r.length_smoothed, r.width_smoothed, color=col, alpha=0.8))
        for _, r in snap.iterrows():
            ahead = snap[(snap.s - r.s > 1) & (snap.s - r.s < 40)]
            if len(ahead):
                j = ahead.iloc[np.argmin(np.hypot(ahead.s - r.s, ahead.d - r.d))]
                ax.annotate("", xy=(j.s, j.d), xytext=(r.s, r.d),
                            arrowprops=dict(arrowstyle="->", lw=0.6, color="0.25"))
        ax.axhline(0, color="0.5", lw=0.5, ls="--")
        ax.set_xlim(S_LO, S_HI)
        ax.set_ylim(-7, 7)
        ax.set_ylabel("$d$ (m)")
        ax.set_title(f"({'ab'[lane - 1]}) Recording 1, direction {lane}, t = {t0:.1f} s "
                     f"({len(snap)} vehicles)", loc="left")
        info[lane] = (t0, len(snap))
    axes[1].set_xlabel("Along-road position $s$ (m), travel to the right")
    fig.tight_layout()
    fig.savefig(FIG / "dataset_snapshot.pdf")
    fig.savefig(FIG / "dataset_snapshot.png", dpi=170)
    plt.close(fig)
    return info


def fmt(x: float, nd: int = 0) -> str:
    return f"{x:,.{nd}f}"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    df = load()
    trips = trajectory_table(df)
    summ = stream_summary(df, trips)
    summ.to_csv(STAT / "dataset_by_stream.csv", index=False)
    print(summ.to_string())

    lines = []
    for r in summ.itertuples():
        lines.append(
            f"{r.run_id} & {r.lane_kf} & {fmt(r.rows)} & {fmt(r.trajectories)} & {fmt(r.light)} & "
            f"{fmt(r.long)} & {fmt(r.short)} & {r.dur_p50_s:.1f} & {r.share_under_5s:.1f} & "
            f"{r.speed_p50:.1f} & {r.speed_p90:.1f} & {r.conc_p50:.0f} \\\\")
    (PAPER / "tables" / "tab_dataset.tex").write_text("\n".join(lines) + "\n")

    conc_rec = {run: df[df.run_id == run].groupby("time").size() for run in (1, 2)}
    by_rec = trips.groupby("run_id")
    dims = trips.groupby(["run_id", "cls"])[["length", "width"]].median()
    w2 = trips[(trips.run_id == 2) & (trips.cls == 2)].width
    macros = {
        "DsRecOneMin": f"{(df[df.run_id == 1].time.max() - df[df.run_id == 1].time.min()) / 60:.1f}",
        "DsRecTwoMin": f"{(df[df.run_id == 2].time.max() - df[df.run_id == 2].time.min()) / 60:.1f}",
        "DsRecOneRows": fmt((df.run_id == 1).sum()),
        "DsRecTwoRows": fmt((df.run_id == 2).sum()),
        "DsRecOneTraj": fmt(by_rec.size()[1]),
        "DsRecTwoTraj": fmt(by_rec.size()[2]),
        "DsConcOne": f"{conc_rec[1].median():.0f}",
        "DsConcTwo": f"{conc_rec[2].median():.0f}",
        "DsConcOneMax": f"{conc_rec[1].max():.0f}",
        "DsConcTwoMax": f"{conc_rec[2].max():.0f}",
        "DsUnderFive": f"{(trips.duration < 5).mean() * 100:.1f}",
        "DsUnderFifteen": f"{(trips.duration < 15).mean() * 100:.1f}",
        "DsLightLenOne": f"{dims.loc[(1, 2), 'length']:.2f}",
        "DsLightLenTwo": f"{dims.loc[(2, 2), 'length']:.2f}",
        "DsLightWidOne": f"{dims.loc[(1, 2), 'width']:.2f}",
        "DsLightWidTwo": f"{dims.loc[(2, 2), 'width']:.2f}",
        "DsLongWidOne": f"{dims.loc[(1, 1), 'width']:.2f}",
        "DsLongWidTwo": f"{dims.loc[(2, 1), 'width']:.2f}",
        "DsLightWidTwoAbove": f"{(w2 > 2.5).mean() * 100:.0f}",
        "DsAccAbove": f"{(df.acceleration_kf.abs() > 4).mean() * 100:.2f}",
        "DsSpeedMax": f"{df.speed_kf.max():.1f}",
        "DsAccMax": f"{df.acceleration_kf.max():.1f}",
        "DsAccMin": f"\\ensuremath{{-}}{abs(df.acceleration_kf.min()):.1f}",
        "DsAbsAccMedian": f"{df.acceleration_kf.abs().median():.2f}",
        "DsAbsAccPninetyfive": f"{df.acceleration_kf.abs().quantile(0.95):.2f}",
        # Times are stored to four decimals, so a multiple of 1/30 s is off by at most 0.0015 frames.
        "DsThirtyfpsShare": f"{(np.abs(df.time * 30 - np.round(df.time * 30)) < 0.005).mean() * 100:.1f}",
    }
    info = fig_snapshot(df)
    macros["DsSnapTimeOne"] = f"{info[1][0]:.1f}"
    macros["DsSnapTimeTwo"] = f"{info[2][0]:.1f}"
    body = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    (PAPER / "dataset_numbers.tex").write_text("% Generated by analysis/dataset_figures.py\n" + body)
    print(body)
    fig_tracks(df, trips)
    fig_timespace(df)
    fig_distributions(df, trips)
    print("figures written")


if __name__ == "__main__":
    main()
