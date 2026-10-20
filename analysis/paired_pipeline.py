#!/usr/bin/env python3
"""Paired SDPM vs RAU experiment.

Stages: prepare, time, stability, pilot.
The cohort file is written only by `prepare` and is not edited after scores exist.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MTM_Car_Calibration import (  # noqa: E402
    compute_mtm_metrics,
    run_ga_mtm,
)
from PT_Car_Calibration import (  # noqa: E402
    compute_car_metrics,
    run_ga_for_id,
)
import highway_geometry  # noqa: E402

DATA = ROOT / "Final_Lebanon_Data.csv"
OUT = ROOT / "outputs"
SEED = 20261004
SAMPLE_EVERY = 5
K_NEIGHBORS = 4
HOLD_FRAC = 0.30

RAU_NAMES = ["Wcc", "eta", "xi", "tau", "v_pref", "Wb"]
SDPM_NAMES = ["v0", "T", "s0", "a", "b", "s0_y", "s0_y_tilde", "lambda", "tau_lat"]


def _circ_mean_deg(ang_deg: np.ndarray) -> float:
    rad = np.radians(ang_deg)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))


def _ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def load_light() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df[df["class"] == 2.0].copy()
    df["run_id"] = df["run_id"].astype(int)
    df["lane_kf"] = df["lane_kf"].astype(int)
    df["id"] = df["id"].astype(int)
    return df


def trip_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, lane, oid), g in df.groupby(["run_id", "lane_kf", "id"], sort=False):
        g = g.sort_values("time")
        x = g.xloc_kf.to_numpy()
        y = g.yloc_kf.to_numpy()
        if len(g) < 5:
            continue
        dx = float(x[-1] - x[0])
        dy = float(y[-1] - y[0])
        dist = float(np.hypot(np.diff(x), np.diff(y)).sum())
        heading = float(np.degrees(np.arctan2(dy, dx))) if math.hypot(dx, dy) > 5 else np.nan
        rows.append(
            {
                "run_id": int(run),
                "lane_kf": int(lane),
                "id": int(oid),
                "n": int(len(g)),
                "duration_s": float(g.time.iloc[-1] - g.time.iloc[0]),
                "path_m": dist,
                "median_speed": float(g.speed_kf.median()),
                "heading_deg": heading,
            }
        )
    return pd.DataFrame(rows)


def prepare() -> None:
    (OUT / "statistics").mkdir(parents=True, exist_ok=True)
    (OUT / "cohorts").mkdir(parents=True, exist_ok=True)
    (OUT / "geometry").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    df = load_light()
    trips = trip_table(df)
    trips.to_csv(OUT / "statistics" / "dataset_integrity.csv", index=False)
    lines = [
        f"light_vehicle_rows {len(df)}",
        f"trajectory_keys {len(trips)}",
        f"streams {trips.groupby(['run_id','lane_kf']).size().to_dict()}",
        "canonical_key run_id,lane_kf,id",
        "class_filter 2",
        "sample_interval_s 0.1",
    ]
    (OUT / "statistics" / "dataset_integrity_summary.txt").write_text("\n".join(lines) + "\n")

    # Stream heading and eligibility. Decided here, before any model score.
    eligible_parts = []
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
        keep["stream_heading_deg"] = med
        eligible_parts.append(keep)
        _write_corridor(df, int(run), int(lane), med)
    eligible = pd.concat(eligible_parts, ignore_index=True)
    rng = random.Random(SEED)
    chosen = []
    for (run, lane), sub in eligible.groupby(["run_id", "lane_kf"]):
        ids = sorted(sub.id.unique())
        take = ids if len(ids) <= 12 else rng.sample(ids, 12)
        take = sorted(take)
        meta = sub.set_index("id")
        for i, oid in enumerate(take):
            row = meta.loc[oid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            chosen.append(
                {
                    "run_id": int(run),
                    "lane_kf": int(lane),
                    "id": int(oid),
                    "canonical_id": f"{int(run)}_{int(lane)}_{int(oid)}",
                    "split": "pilot" if i < 6 else "reserve",
                    "n": int(row.n),
                    "duration_s": float(row.duration_s),
                    "path_m": float(row.path_m),
                    "median_speed": float(row.median_speed),
                    "heading_deg": float(row.heading_deg),
                }
            )
    cohort = pd.DataFrame(chosen)
    cohort_path = OUT / "cohorts" / "paired_cohort.csv"
    if cohort_path.exists():
        print("cohort already exists; not rewriting", cohort_path)
    else:
        cohort.to_csv(cohort_path, index=False)
        print("wrote", cohort_path, "rows", len(cohort))
    _parameter_dictionary()
    highway_geometry.load_envelope_boundaries()
    print("corridors", list(highway_geometry._CORRIDORS))


def _write_corridor(df: pd.DataFrame, run: int, lane: int, heading_med: float) -> None:
    sub = df[(df.run_id == run) & (df.lane_kf == lane)]
    # Keep points from trips aligned with the stream. This uses the whole
    # stream, not one focal vehicle's future.
    heads = []
    for oid, g in sub.groupby("id"):
        g = g.sort_values("time")
        if len(g) < 5:
            continue
        dx = float(g.xloc_kf.iloc[-1] - g.xloc_kf.iloc[0])
        dy = float(g.yloc_kf.iloc[-1] - g.yloc_kf.iloc[0])
        if math.hypot(dx, dy) < 5:
            continue
        h = math.degrees(math.atan2(dy, dx))
        if abs(_ang_diff(h, heading_med)) <= 25:
            heads.append(g)
    if not heads:
        return
    pts = pd.concat(heads, ignore_index=True)
    xy = pts[["xloc_kf", "yloc_kf"]].to_numpy(dtype=float)
    mu = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - mu, full_matrices=False)
    u = vt[0]
    # Orient the axis with the stream heading.
    hd = math.radians(heading_med)
    if u[0] * math.cos(hd) + u[1] * math.sin(hd) < 0:
        u = -u
    s = (xy - mu) @ u
    order = np.argsort(s)
    # Bin medians, 15 m.
    s_min, s_max = float(s.min()), float(s.max())
    edges = np.arange(s_min, s_max + 15.0, 15.0)
    centers = []
    for i in range(len(edges) - 1):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if m.sum() < 40:
            continue
        centers.append(np.median(xy[m], axis=0))
    if len(centers) < 4:
        return
    c = np.asarray(centers, dtype=float)
    # One smoothing pass, endpoints fixed.
    sm = c.copy()
    for i in range(1, len(c) - 1):
        sm[i] = (c[i - 1] + c[i] + c[i + 1]) / 3.0
    # Signed lateral offset to the nearest centerline segment.
    # A nearest-vertex dot product leaks along-track error on a curve.
    a = sm[:-1]
    b = sm[1:]
    ab = b - a
    ab2 = np.maximum(np.sum(ab * ab, axis=1), 1e-9)
    ap = xy[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / ab2[None, :], 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    diff = xy[:, None, :] - proj
    j = np.argmin(np.sum(diff * diff, axis=2), axis=1)
    chosen = diff[np.arange(len(xy)), j]
    chosen_ab = ab[j]
    cross = chosen_ab[:, 0] * chosen[:, 1] - chosen_ab[:, 1] * chosen[:, 0]
    lat = cross / np.sqrt(ab2[j])
    # Drop gross outliers (loops and tracking hooks) before the percentile.
    keep = np.abs(lat) <= 15.0
    lo_s = np.full(len(sm) - 1, np.nan)
    hi_s = np.full(len(sm) - 1, np.nan)
    for i in range(len(sm) - 1):
        m = keep & (j == i)
        if int(m.sum()) < 30:
            continue
        lo_s[i] = float(np.quantile(lat[m], 0.05))
        hi_s[i] = float(np.quantile(lat[m], 0.95))
    for arr in (lo_s, hi_s):
        good = np.isfinite(arr)
        if good.any() and not good.all():
            idx = np.arange(len(arr))
            arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    lo = np.empty(len(sm))
    hi = np.empty(len(sm))
    lo[0], hi[0] = lo_s[0], hi_s[0]
    lo[-1], hi[-1] = lo_s[-1], hi_s[-1]
    if len(sm) > 2:
        lo[1:-1] = 0.5 * (lo_s[:-1] + lo_s[1:])
        hi[1:-1] = 0.5 * (hi_s[:-1] + hi_s[1:])
    tang = np.zeros_like(sm)
    tang[:-1] = ab
    tang[-1] = ab[-1]
    for i in range(1, len(tang)):
        if tang[i] @ tang[i - 1] < 0:
            tang[i] *= -1
    nrm = np.linalg.norm(tang, axis=1)
    nrm[nrm < 1e-6] = 1.0
    tang = tang / nrm[:, None]
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    # Fill empty bins from neighbors.
    for arr in (lo, hi):
        good = np.isfinite(arr)
        if good.any():
            idx = np.arange(len(arr))
            arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    x_lo = sm[:, 0] + lo * normal[:, 0]
    y_lo = sm[:, 1] + lo * normal[:, 1]
    x_hi = sm[:, 0] + hi * normal[:, 0]
    y_hi = sm[:, 1] + hi * normal[:, 1]
    arc = np.zeros(len(sm))
    arc[1:] = np.cumsum(np.hypot(np.diff(sm[:, 0]), np.diff(sm[:, 1])))
    out = np.column_stack(
        [arc, sm[:, 0], sm[:, 1], normal[:, 0], normal[:, 1], x_lo, y_lo, x_hi, y_hi]
    )
    path = OUT / "geometry" / f"run{run}_dir{lane}_corridor.csv"
    header = "s,x,y,nx,ny,x_lo,y_lo,x_hi,y_hi"
    np.savetxt(path, out, delimiter=",", header=header, comments="")
    width = float(np.median(hi - lo))
    print(f"corridor run {run} dir {lane} vertices {len(sm)} median width {width:.2f} m")


def _parameter_dictionary() -> None:
    rau = pd.DataFrame(
        [
            ("Wcc", "W_cc", "conflict penalty weight", "1", 10, 5000),
            ("eta", "eta", "progress scale", "1", 0.1, 10),
            ("xi", "xi", "speed-curvature exponent", "1", 0.1, 15),
            ("tau", "tau", "execution-noise relaxation", "s", 0.1, 3),
            ("v_pref", "v_pref", "preferred speed", "m/s", 1, 50),
            ("Wb", "W_b", "corridor-boundary penalty", "1", 0, 10000),
        ],
        columns=["code", "symbol", "meaning", "units", "lower", "upper"],
    )
    sdpm = pd.DataFrame(
        [
            ("v0", "v0", "desired speed in the IDM core", "m/s", 5, 40),
            ("T", "T", "desired time gap", "s", 0.2, 4),
            ("s0", "s0", "minimum gap", "m", 0.5, 6),
            ("a", "a", "maximum acceleration", "m/s^2", 0.3, 4),
            ("b", "b", "comfortable deceleration", "m/s^2", 0.3, 5),
            ("s0_y", "s0_y", "lateral attenuation scale", "m", 0.05, 1),
            ("s0_y_tilde", "s0_y_tilde", "lateral gap scale", "m", 0.05, 1.5),
            ("lambda", "lambda", "lateral sensitivity", "s", 0.05, 2),
            ("tau_lat", "tau_lat", "lateral relaxation time", "s", 0.2, 5),
        ],
        columns=["code", "symbol", "meaning", "units", "lower", "upper"],
    )
    rau.to_csv(OUT / "statistics" / "rau_parameter_dictionary.csv", index=False)
    sdpm.to_csv(OUT / "statistics" / "sdpm_parameter_dictionary.csv", index=False)


def _stream_index(df: pd.DataFrame):
    """time -> ndarray [id, x, y, vx, vy, length, width] per stream."""
    out = {}
    for (run, lane), sub in df.groupby(["run_id", "lane_kf"]):
        pieces = []
        for oid, g in sub.groupby("id", sort=False):
            g = g.sort_values("time")
            t = g.time.to_numpy(dtype=float)
            x = g.xloc_kf.to_numpy(dtype=float)
            y = g.yloc_kf.to_numpy(dtype=float)
            dt = np.diff(t)
            vx = np.zeros_like(x)
            vy = np.zeros_like(y)
            ok = dt > 1e-6
            vx[1:][ok] = np.diff(x)[ok] / dt[ok]
            vy[1:][ok] = np.diff(y)[ok] / dt[ok]
            if len(vx) > 1:
                vx[0] = vx[1]
                vy[0] = vy[1]
            pieces.append(
                np.column_stack(
                    [
                        np.full(len(g), oid),
                        x,
                        y,
                        vx,
                        vy,
                        g.length_smoothed.to_numpy(dtype=float),
                        g.width_smoothed.to_numpy(dtype=float),
                        np.round(t, 1),
                    ]
                )
            )
        arr = np.vstack(pieces)
        by_t = {}
        for tk in np.unique(arr[:, 7]):
            by_t[float(tk)] = arr[arr[:, 7] == tk][:, :7]
        out[(int(run), int(lane))] = by_t
    return out


def build_frames(index, run, lane, oid) -> dict:
    by_t = index[(run, lane)]
    # Focal samples.
    focal_times = []
    focal_rows = []
    for tk in sorted(by_t):
        block = by_t[tk]
        m = block[:, 0] == oid
        if not np.any(m):
            continue
        focal_rows.append(block[m][0])
        focal_times.append(tk)
    if len(focal_times) < 8:
        return {}
    sel = list(range(0, len(focal_times), SAMPLE_EVERY))
    frames = {}
    dest = (float(focal_rows[-1][1]), float(focal_rows[-1][2]))
    for j in sel:
        tk = focal_times[j]
        ego = focal_rows[j]
        block = by_t[tk]
        others = block[block[:, 0] != oid]
        if len(others):
            d = np.hypot(others[:, 1] - ego[1], others[:, 2] - ego[2])
            k = min(K_NEIGHBORS, len(others))
            take = np.argpartition(d, k - 1)[:k]
            nbs = [
                (
                    float(others[i, 1]),
                    float(others[i, 2]),
                    float(others[i, 3]),
                    float(others[i, 4]),
                    float(others[i, 5]),
                    float(others[i, 6]),
                )
                for i in take
            ]
        else:
            nbs = []
        frames[tk] = [
            {
                "id": int(oid),
                "x": float(ego[1]),
                "y": float(ego[2]),
                "vx": float(ego[3]),
                "vy": float(ego[4]),
                "dest_x": dest[0],
                "dest_y": dest[1],
                "neighbors": nbs,
                "lane_kf": float(lane),
                "run_id": float(run),
                "length_m": float(ego[5]),
                "width_m": float(ego[6]),
            }
        ]
    return frames


def _split(frames: dict):
    times = sorted(frames)
    cut = max(4, int(round(len(times) * (1.0 - HOLD_FRAC))))
    cut = min(cut, len(times) - 3)
    train = {t: frames[t] for t in times[:cut]}
    test = {t: frames[t] for t in times[cut:]}
    return train, test


def _ga(kind: str, frames, oid, pop, gen, seed):
    """Existing crossover GA. Search seed is explicit so two runs can differ."""
    if kind == "RAU":
        vec, fit, _, hist = run_ga_for_id(
            frames,
            oid,
            pop_size=pop,
            generations=gen,
            n_workers=1,
            clip_to_envelope=True,
            search_seed=seed,
        )
    else:
        vec, fit, _, hist = run_ga_mtm(
            frames,
            oid,
            pop_size=pop,
            generations=gen,
            n_workers=1,
            clip_to_envelope=True,
            search_seed=seed,
        )
    return list(vec), float(fit), list(hist)


def _metrics(kind, frames, oid, params, rng=None) -> dict:
    empty = {
        "status": "too_short",
        "ADE": math.nan,
        "FDE": math.nan,
        "RMSE_s": math.nan,
        "RMSE_d": math.nan,
        "RMSE_speed": math.nan,
        "loss": math.nan,
    }
    if not frames or len(frames) < 2:
        return empty
    if kind == "RAU":
        m = compute_car_metrics(frames, params, oid, clip_to_envelope=True, rng=rng)
    else:
        m = compute_mtm_metrics(
            frames,
            params,
            oid,
            clip_to_envelope=True,
            rng=rng,
        )
    ade = float(m["ADE"])
    finite = math.isfinite(ade) and ade < 1e6
    loss = (1.0 / m["Error"] - 1.0) if m["Error"] > 1e-9 else 1e9
    u0 = next(iter(frames.values()))[0]
    if "sim_x" in m and finite:
        rmse_s, rmse_d, rmse_v = highway_geometry.road_frame_rmse(
            m["sim_x"],
            m["sim_y"],
            m["obs_x"],
            m["obs_y"],
            m["sim_speed"],
            m["obs_speed"],
            u0["run_id"],
            u0["lane_kf"],
        )
    else:
        rmse_s = rmse_d = rmse_v = math.nan
    return {
        "status": "ok" if finite else "nonfinite",
        "ADE": ade,
        "FDE": float(m["FDE"]),
        "RMSE": float(m["RMSE"]),
        "RMSE_s": rmse_s,
        "RMSE_d": rmse_d,
        "RMSE_speed": rmse_v,
        "loss": float(loss),
        "Error": float(m["Error"]),
    }


def _load_index():
    cache = OUT / "cache" / "stream_index.pkl"
    highway_geometry.load_envelope_boundaries()
    if cache.exists():
        print("loading cached index", flush=True)
        with cache.open("rb") as fh:
            return pickle.load(fh)
    print("loading trajectories", flush=True)
    df = load_light()
    print("indexing streams", flush=True)
    t0 = time.time()
    index = _stream_index(df)
    print(f"index {time.time() - t0:.1f}s", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump(index, fh, protocol=4)
    return index


def time_one():
    cohort = pd.read_csv(OUT / "cohorts" / "paired_cohort.csv")
    row = cohort[cohort.split == "pilot"].iloc[0]
    index = _load_index()
    frames = build_frames(index, int(row.run_id), int(row.lane_kf), int(row.id))
    train, test = _split(frames)
    print("train frames", len(train), "test", len(test), "id", row.canonical_id)
    p = clamp_car_params(random_car_params())
    t0 = time.time()
    compute_car_metrics(train, p, int(row.id), clip_to_envelope=True)
    dt_r = time.time() - t0
    q = clamp_mtm_params(random_mtm_params())
    t0 = time.time()
    compute_mtm_metrics(
        train,
        q,
        int(row.id),
        clip_to_envelope=True,
    )
    dt_s = time.time() - t0
    print(f"one RAU eval {dt_r:.3f}s  one SDPM eval {dt_s:.3f}s")
    # Repeat RAU to confirm identical score.
    a = compute_car_metrics(train, p, int(row.id))["Error"]
    b = compute_car_metrics(train, p, int(row.id))["Error"]
    print("RAU repeat", a, b, "match", abs(a - b) < 1e-12)


def _stability_job(job):
    """One GA on one track. Runs in a worker process."""
    kind, frames, oid, pop, gen, seed, canonical_id = job
    highway_geometry.load_envelope_boundaries()
    t0 = time.time()
    try:
        vec, fit, hist = _ga(kind, frames, int(oid), int(pop), int(gen), int(seed))
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        vec, fit, hist, status = [], math.nan, [], str(exc)[:160]
    return {
        "canonical_id": canonical_id,
        "model": kind,
        "pop": int(pop),
        "gen": int(gen),
        "seed": int(seed),
        "fitness": fit,
        "final_loss": (1.0 / fit - 1.0) if isinstance(fit, float) and fit > 1e-9 else math.nan,
        "seconds": time.time() - t0,
        "status": status,
        "history": hist,
        "n_params": len(vec),
    }


def stability():
    """Four pilot tracks, two seeds, three budgets. Writes frozen_budget.json."""
    cohort = pd.read_csv(OUT / "cohorts" / "paired_cohort.csv")
    # One pilot trajectory from each recording-direction stream.
    # head(4) would take four tracks from the first stream only.
    pilot = (
        cohort[cohort.split == "pilot"]
        .groupby(["run_id", "lane_kf"], sort=True)
        .head(1)
        .reset_index(drop=True)
    )
    index = _load_index()
    budgets = [(10, 20), (20, 40), (30, 60)]
    jobs = []
    for _, row in pilot.iterrows():
        frames = build_frames(index, int(row.run_id), int(row.lane_kf), int(row.id))
        train, _ = _split(frames)
        for pop, gen in budgets:
            for seed_i in (0, 1):
                seed = SEED + seed_i * 1000 + int(row.id)
                for kind in ("SDPM", "RAU"):
                    jobs.append((kind, train, int(row.id), pop, gen, seed, row.canonical_id))
    workers = min(8, os.cpu_count() or 2)
    print(f"stability jobs {len(jobs)} workers {workers}", flush=True)
    rows = []
    hist_rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_stability_job, job) for job in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            hist = rec.pop("history")
            rows.append(rec)
            for g, loss in enumerate(hist, start=1):
                hist_rows.append(
                    {
                        "canonical_id": rec["canonical_id"],
                        "model": rec["model"],
                        "pop": rec["pop"],
                        "gen": rec["gen"],
                        "seed": rec["seed"],
                        "generation": g,
                        "loss": loss,
                    }
                )
            print(
                rec["model"],
                rec["canonical_id"],
                f"pop {rec['pop']} gen {rec['gen']}",
                "fit",
                None if rec["fitness"] != rec["fitness"] else round(rec["fitness"], 4),
                f"{rec['seconds']:.1f}s",
                rec["status"],
                flush=True,
            )
    out = pd.DataFrame(rows)
    path = OUT / "statistics" / "optimizer_stability.csv"
    out.to_csv(path, index=False)
    pd.DataFrame(hist_rows).to_csv(OUT / "statistics" / "optimizer_history.csv", index=False)
    summary = out.groupby(["model", "pop", "gen"])["fitness"].median()
    print(summary)
    pop, gen = _choose_budget(out)
    (OUT / "statistics" / "frozen_budget.json").write_text(
        json.dumps(
            {
                "pop": pop,
                "gen": gen,
                "rule": "smallest budget unless the next larger one gains 0.02 median fitness or 0.15 worst-run fitness",
                "source": "optimizer_stability.csv",
            }
        )
    )
    print("frozen", pop, gen)


def _choose_budget(stab: pd.DataFrame) -> tuple[int, int]:
    """Walk budgets from smallest to largest.

    Keep a larger budget if median fitness rises by at least 0.02, or if the
    worst run rises by at least 0.15. The second clause is there because a
    discarded first pass, on a folded boundary, had a median that hid a seed
    stuck near fitness 0.03.
    """
    med = stab.groupby(["pop", "gen"])["fitness"].median()
    worst = stab.groupby(["pop", "gen"])["fitness"].min()
    keys = sorted(med.index, key=lambda k: int(k[0]) * int(k[1]))
    if not keys:
        return 10, 20
    chosen = keys[0]
    for nxt in keys[1:]:
        gain = float(med.loc[nxt] - med.loc[chosen])
        tail = float(worst.loc[nxt] - worst.loc[chosen])
        if gain >= 0.02 or tail >= 0.15:
            chosen = nxt
        else:
            break
    return int(chosen[0]), int(chosen[1])


def _empty_metrics():
    return {
        "ADE": math.nan,
        "FDE": math.nan,
        "loss": math.nan,
        "RMSE_s": math.nan,
        "RMSE_d": math.nan,
        "RMSE_speed": math.nan,
        "status": "error",
    }


def _pilot_one(payload):
    """Calibrate both models on one trajectory. One search seed per trajectory."""
    highway_geometry.load_envelope_boundaries()
    train = payload["train"]
    test = payload["test"]
    meta = payload["meta"]
    pop, gen, seed = payload["pop"], payload["gen"], payload["seed"]
    oid = int(meta["id"])
    rec = dict(meta)
    rec["pop"] = pop
    rec["gen"] = gen
    rec["search_seed"] = seed
    for kind, names in (("SDPM", SDPM_NAMES), ("RAU", RAU_NAMES)):
        t0 = time.time()
        try:
            vec, fit, _hist = _ga(kind, train, oid, pop, gen, seed)
            tr = _metrics(kind, train, oid, vec)
            te = _metrics(kind, test, oid, vec)
            status = te["status"]
        except Exception as exc:  # noqa: BLE001
            vec = [math.nan] * len(names)
            fit = math.nan
            tr = _empty_metrics()
            te = _empty_metrics()
            status = str(exc)[:160]
        rec[f"{kind}_train_fitness"] = fit
        rec[f"{kind}_train_loss"] = tr["loss"]
        rec[f"{kind}_train_ADE"] = tr["ADE"]
        rec[f"{kind}_train_FDE"] = tr["FDE"]
        rec[f"{kind}_train_RMSE_s"] = tr.get("RMSE_s", math.nan)
        rec[f"{kind}_train_RMSE_d"] = tr.get("RMSE_d", math.nan)
        rec[f"{kind}_train_RMSE_speed"] = tr.get("RMSE_speed", math.nan)
        rec[f"{kind}_holdout_ADE"] = te["ADE"]
        rec[f"{kind}_holdout_FDE"] = te["FDE"]
        rec[f"{kind}_holdout_RMSE_s"] = te.get("RMSE_s", math.nan)
        rec[f"{kind}_holdout_RMSE_d"] = te.get("RMSE_d", math.nan)
        rec[f"{kind}_holdout_RMSE_speed"] = te.get("RMSE_speed", math.nan)
        rec[f"{kind}_holdout_loss"] = te["loss"]
        rec[f"{kind}_seconds"] = time.time() - t0
        rec[f"{kind}_status"] = status
        for name, val in zip(names, vec):
            rec[f"{kind}_{name}"] = val
    for metric in ("ADE", "FDE", "RMSE_s", "RMSE_d", "RMSE_speed"):
        a = rec.get(f"RAU_holdout_{metric}", math.nan)
        b = rec.get(f"SDPM_holdout_{metric}", math.nan)
        if isinstance(a, float) and isinstance(b, float) and math.isfinite(a) and math.isfinite(b):
            rec[f"delta_{metric}"] = a - b
    return rec


def _calibrate_splits(splits, out_name, seed_offset):
    stab_path = OUT / "statistics" / "optimizer_stability.csv"
    if stab_path.exists():
        pop, gen = _choose_budget(pd.read_csv(stab_path))
    else:
        pop, gen = 20, 40
    (OUT / "statistics" / "frozen_budget.json").write_text(
        json.dumps(
            {
                "pop": pop,
                "gen": gen,
                "source": "optimizer_stability.csv" if stab_path.exists() else "default",
                "rule": "smallest budget unless the next larger one gains 0.02 median fitness or 0.15 worst-run fitness",
            }
        )
    )
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(OUT / "cohorts" / "paired_cohort.csv")
    pilot_df = cohort[cohort.split.isin(splits)].reset_index(drop=True)
    index = _load_index()
    jobs = []
    for i, row in pilot_df.iterrows():
        frames = build_frames(index, int(row.run_id), int(row.lane_kf), int(row.id))
        train, test = _split(frames)
        meta = {
            "canonical_id": row.canonical_id,
            "split": row.split,
            "run_id": int(row.run_id),
            "lane_kf": int(row.lane_kf),
            "id": int(row.id),
            "duration_s": float(row.duration_s),
            "path_m": float(row.path_m),
            "median_speed": float(row.median_speed),
            "n_train_frames": len(train),
            "n_holdout_frames": len(test),
        }
        jobs.append(
            {
                "train": train,
                "test": test,
                "meta": meta,
                "pop": pop,
                "gen": gen,
                "seed": SEED + seed_offset + int(i),
            }
        )
    workers = min(8, os.cpu_count() or 2)
    print(f"jobs {len(jobs)} splits {splits} pop {pop} gen {gen} workers {workers}", flush=True)
    rows = []
    out_path = OUT / "results" / out_name
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_pilot_one, job) for job in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(
                rec["canonical_id"],
                "SDPM ADE",
                rec.get("SDPM_holdout_ADE"),
                "RAU ADE",
                rec.get("RAU_holdout_ADE"),
                "delta",
                rec.get("delta_ADE"),
                flush=True,
            )
    print("rows", len(rows), "wrote", out_path)


def pilot():
    _calibrate_splits(["pilot"], "pilot_paired_results.csv", 0)


def extended():
    """Reserve and main splits. Same budget and metrics as the pilot. Seed offset 5000."""
    _calibrate_splits(["reserve", "main"], "extended_paired_results.csv", 5000)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if cmd == "prepare":
        prepare()
    elif cmd == "time":
        time_one()
    elif cmd == "stability":
        stability()
    elif cmd == "pilot":
        pilot()
    elif cmd == "extended":
        extended()
    else:
        raise SystemExit(f"unknown stage {cmd}")
