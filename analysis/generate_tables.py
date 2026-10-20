#!/usr/bin/env python3
"""LaTeX tables and number macros for the manuscript, read from the result CSVs.

Every number quoted in the text or tables is defined here, so the manuscript
cannot drift from the outputs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paired_statistics import _boot_ci

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs" / "results"
STAT = ROOT / "outputs" / "statistics"
TAB = ROOT / "outputs" / "tables"
PAPER = ROOT / "paper"
PAPER_TABLES = PAPER / "tables"
LEGACY_TAB = PAPER / "analysis_outputs" / "tables"

METRIC_TEX = {
    "ADE": r"ADE (m)",
    "RMSE_s": r"RMSE$_s$ (m)",
    "RMSE_d": r"RMSE$_d$ (m)",
    "RMSE_speed": r"RMSE$_v$ (m/s)",
    "FDE": r"FDE (m)",
}
SYMBOL_TEX = {
    "v0": r"$v_0$", "T": r"$T$", "s0": r"$s_0$", "a": r"$a$", "b": r"$b$",
    "s0_y": r"$s_0^y$", "s0_y_tilde": r"$\tilde s_0^y$", "lambda": r"$\lambda$",
    "tau_lat": r"$\tau_{\mathrm{lat}}$",
    "Wcc": r"$W_{cc}$", "eta": r"$\eta$", "xi": r"$\xi$", "tau": r"$\tau$",
    "v_pref": r"$v_{\mathrm{pref}}$", "Wb": r"$W_b$",
}
UNIT_TEX = {"m/s": r"m\,s$^{-1}$", "m/s^2": r"m\,s$^{-2}$", "s": "s", "m": "m", "--": "--", "—": "--"}

macros: dict[str, str] = {}
RUN_WORD = {1: "One", 2: "Two"}


def f(x, nd=1):
    return f"{x:.{nd}f}"


def p_fmt(p):
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def signed(x, nd=1):
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        return s.lstrip("-")
    return s.replace("-", r"\ensuremath{-}")


def ci(lo, hi, nd=1):
    return f"[{signed(lo, nd)}, {signed(hi, nd)}]"


def m(name, value):
    macros[name] = value


def write(name, body):
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    (PAPER_TABLES / f"{name}.tex").write_text(body)


def data_table():
    di = pd.read_csv(STAT / "dataset_integrity.csv").set_index("statistic")["value"]
    lt = pd.read_csv(LEGACY_TAB / "lateral_and_leaders.csv")
    m("RawRows", f"{int(di.raw_rows):,}".replace(",", "{,}"))
    m("NTrajAll", f"{int(di.unique_canonical_trajectories):,}".replace(",", "{,}"))
    n_car = int(sum(di[f"trajectories_class2_run{r}_dir{d}"] for r in (1, 2) for d in (1, 2)))
    m("NTrajCar", f"{n_car:,}".replace(",", "{,}"))
    m("IdsReusedRun", str(int(di.ids_reused_across_run_id)))
    m("IdsReusedDir", str(int(di.ids_reused_across_lane_kf)))
    m("DurMedian", f(di.duration_s_p50))
    m("PathMedian", f(di.path_m_p50, 0))
    lines = []
    for r in (1, 2):
        for d in (1, 2):
            row = lt[(lt.run_id == r) & (lt.lane_kf == d)].iloc[0]
            lines.append(
                f"{r} & {d} & {int(di[f'trajectories_run{r}_dir{d}']):,} & "
                f"{int(di[f'trajectories_class2_run{r}_dir{d}']):,} & "
                f"{f(row.leader_abs_d_p50, 2)} & {100 * row.frac_leader_abs_d_gt_1p5m:.1f} \\\\"
            )
    write("tab_data", "\n".join(lines) + "\n")


def stagger():
    pooled = pd.read_csv(LEGACY_TAB / "leader_offset_pooled.csv").iloc[0]
    thr = pd.read_csv(TAB / "threshold_sensitivity.csv")
    m("StagN", f"{int(pooled.n_pairs):,}".replace(",", "{,}"))
    m("StagMedian", f(pooled.leader_abs_d_p50, 1))
    m("StagPninety", f(pooled.leader_abs_d_p90, 1))
    m("StagGap", f(pooled.gap_p50, 1))

    def at(d):
        return 100 * float(thr.loc[(thr.delta_m - d).abs().idxmin(), "frac_outside"])

    m("StagOne", f(at(1.0)))
    m("StagOneFive", f(at(1.5)))
    m("StagTwo", f(at(2.0)))
    m("StagThree", f(at(3.0)))
    m("StagHalf", f(at(0.5)))
    ad = np.load(STAT / "nearest_forward_abs_offset.npy")
    m("StagInline", f(100 * np.mean(ad < 0.5)))
    m("StagMid", f(100 * np.mean((ad >= 1.0) & (ad < 2.5))))
    dens, edges = np.histogram(ad, bins=np.arange(0, 6.01, 0.25), density=True)
    lo_mode = int(np.argmax(dens[:4]))
    hi_mode = 6 + int(np.argmax(dens[6:]))
    trough = lo_mode + int(np.argmin(dens[lo_mode:hi_mode + 1]))
    m("StagModeHiLo", f(edges[hi_mode], 2))
    m("StagModeHiHi", f(edges[hi_mode + 1], 2))
    m("StagTroughLo", f(edges[trough], 2))
    m("StagTroughHi", f(edges[trough + 1], 2))
    m("StagTroughRatio", f(dens[trough] / dens[hi_mode], 2))
    rows = [f"{d:.1f} & {at(d):.1f} \\\\" for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    body = (
        "\\begin{tabular}{cc}\n\\toprule\n$\\delta$ (m) & Pairs with $|\\Delta d|>\\delta$ (\\%) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (TAB / "threshold_sensitivity.tex").write_text(body)


def comparison(df):
    pt = pd.read_csv(STAT / "paired_tests.csv")
    lines = []
    for _, r in pt.iterrows():
        lines.append(
            f"{METRIC_TEX[r.metric]} & {f(r.sdpm_median)} & {f(r.rau_median)} & "
            f"{signed(r.delta_median)} {ci(r.ci95_median_low, r.ci95_median_high)} & "
            f"{signed(r.delta_mean)} {ci(r.delta_mean_ci_low, r.delta_mean_ci_high)} & "
            f"{p_fmt(r.p_wilcoxon)} & {p_fmt(r.p_holm)} & {signed(r.rank_biserial, 2)} & "
            f"{int(r.n_rau_better)} \\\\"
        )
        key = r.metric.replace("_", "").replace("RMSEspeed", "RMSEv")
        m(f"Sdpm{key}", f(r.sdpm_median))
        m(f"Rau{key}", f(r.rau_median))
        m(f"Dmed{key}", signed(r.delta_median))
        m(f"Dlo{key}", signed(r.ci95_median_low))
        m(f"Dhi{key}", signed(r.ci95_median_high))
        m(f"Dmean{key}", signed(r.delta_mean))
        m(f"Dmeanlo{key}", signed(r.delta_mean_ci_low))
        m(f"Dmeanhi{key}", signed(r.delta_mean_ci_high))
        m(f"P{key}", p_fmt(r.p_wilcoxon))
        m(f"Pholm{key}", p_fmt(r.p_holm))
        m(f"Rrb{key}", signed(r.rank_biserial, 2))
        m(f"Nrau{key}", str(int(r.n_rau_better)))
        m(f"Nsdpm{key}", str(int(r.n_sdpm_better)))
    body = "\n".join(lines) + "\n"
    write("tab_paired", body)
    full = (
        "\\begin{tabular}{lcccccccc}\n\\toprule\n"
        "Metric & SDPM median & RAU median & Median $\\Delta$ [95\\% CI] & Mean $\\Delta$ [95\\% CI] & "
        "$p_W$ & $p_{\\mathrm{Holm}}$ & $r_{rb}$ & RAU lower \\\\\n\\midrule\n"
        + body + "\\bottomrule\n\\end{tabular}\n"
    )
    (TAB / "final_model_comparison.tex").write_text(full)
    pt.to_csv(TAB / "final_model_comparison.csv", index=False)
    m("NPaired", str(int(pt.n.iloc[0])))
    m("NFail", str(int((df.SDPM_status != "ok").sum() + (df.RAU_status != "ok").sum())))
    m("SdpmSecMedian", f(df.SDPM_seconds.median(), 0))
    m("RauSecMedian", f(df.RAU_seconds.median(), 0))


def streams(df):
    lines = []
    for (r, d), sub in df.groupby(["run_id", "lane_kf"]):
        delta = sub.delta_ADE.to_numpy()
        lo, hi = _boot_ci(delta, np.median)
        lines.append(
            f"{r} & {d} & {len(sub)} & {f(sub.SDPM_holdout_ADE.median())} & {f(sub.RAU_holdout_ADE.median())} & "
            f"{signed(np.median(delta))} {ci(lo, hi)} & {int((delta < 0).sum())} \\\\"
        )
        m(f"StreamDmed{'ABCD'[(r - 1) * 2 + d - 1]}", signed(np.median(delta)))
        m(f"StreamNrau{'ABCD'[(r - 1) * 2 + d - 1]}", str(int((delta < 0).sum())))
    write("tab_stream", "\n".join(lines) + "\n")
    # held-out splits
    for split, sub in df.groupby("split"):
        delta = sub.delta_ADE.to_numpy()
        lo, hi = _boot_ci(delta, np.median)
        tag = split.capitalize()
        m(f"Split{tag}N", str(len(sub)))
        m(f"Split{tag}Dmed", signed(np.median(delta)))
        m(f"Split{tag}Dlo", signed(lo))
        m(f"Split{tag}Dhi", signed(hi))


def horizon():
    h = pd.read_csv(TAB / "horizon_error.csv")
    lines = []
    for _, r in h.iterrows():
        lines.append(
            f"{int(r.horizon_s)} & {int(r.n)} & {f(r.SDPM_median_m)} [{f(r.SDPM_q25)}, {f(r.SDPM_q75)}] & "
            f"{f(r.RAU_median_m)} [{f(r.RAU_q25)}, {f(r.RAU_q75)}] & "
            f"{signed(r.delta_median_m)} {ci(r.ci95_low, r.ci95_high)} & {p_fmt(r.p_wilcoxon)} & "
            f"{signed(r.rank_biserial, 2)} \\\\"
        )
        k = {2: "Two", 5: "Five", 10: "Ten", 15: "Fifteen"}[int(r.horizon_s)]
        m(f"HorN{k}", str(int(r.n)))
        m(f"HorSdpm{k}", f(r.SDPM_median_m))
        m(f"HorRau{k}", f(r.RAU_median_m))
        m(f"HorD{k}", signed(r.delta_median_m))
        m(f"HorDlo{k}", signed(r.ci95_low))
        m(f"HorDhi{k}", signed(r.ci95_high))
    body = "\n".join(lines) + "\n"
    write("tab_horizon", body)
    full = (
        "\\begin{tabular}{ccccccc}\n\\toprule\n"
        "Horizon (s) & $N$ & SDPM median [IQR] (m) & RAU median [IQR] (m) & Median $\\Delta$ [95\\% CI] (m) & "
        "$p_W$ & $r_{rb}$ \\\\\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n"
    )
    (TAB / "horizon_error.tex").write_text(full)
    rc = pd.read_csv(STAT / "replay_check.csv")
    m("HoldDurMedian", f(rc.holdout_duration_s.median()))
    m("HoldDurMin", f(rc.holdout_duration_s.min()))
    m("HoldDurMax", f(rc.holdout_duration_s.max()))


def bounds(df):
    b = pd.read_csv(TAB / "parameter_bounds.csv")
    lines = []
    prev = None
    for _, r in b.iterrows():
        model = r.model if r.model != prev else ""
        if prev is not None and r.model != prev:
            lines.append("\\midrule")
        prev = r.model
        lo = f"{r.lower:g}"
        hi = f"{r.upper:g}"
        lines.append(
            f"{model} & {SYMBOL_TEX[r.parameter]} & {r.meaning} & {UNIT_TEX[r.units]} & {lo} & {hi} & "
            f"{r['median']:.3g} [{r.iqr:.3g}] & {100 * r.frac_within_5pct_lower:.0f} & "
            f"{100 * r.frac_within_5pct_upper:.0f} \\\\"
        )
    body = "\n".join(lines) + "\n"
    write("tab_params", body)
    full = (
        "\\begin{tabular}{llllrrrrr}\n\\toprule\n"
        "Model & Symbol & Meaning & Units & Lower & Upper & Median [IQR] & \\% near lower & \\% near upper \\\\\n"
        "\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n"
    )
    (TAB / "parameter_bounds.tex").write_text(full)
    for model in ("SDPM", "RAU"):
        sub = b[b.model == model]
        z = []
        for _, r in sub.iterrows():
            v = df[f"{model}_{r.parameter}"].to_numpy(dtype=float)
            z.append((v - r.lower) / (r.upper - r.lower))
        z = np.column_stack(z)
        near = (z <= 0.05) | (z >= 0.95)
        m(f"{model.capitalize()}AnyBound", f"{100 * near.any(axis=1).mean():.0f}")
    m("WbNearZero", str(int((df.RAU_Wb < 1.0).sum())))
    m("WccAtLower", str(int((df.RAU_Wcc < 15.0).sum())))
    m("TauAtLower", str(int((df.RAU_tau < 0.15).sum())))
    m("WbMedian", f(df.RAU_Wb.median(), 0))
    m("WccMedian", f(df.RAU_Wcc.median(), 0))
    m("WbPninetyfive", f(df.RAU_Wb.quantile(0.95), 0))
    m("WccPninetyfive", f(df.RAU_Wcc.quantile(0.95), 0))
    for _, r in b.iterrows():
        key = f"{r.model.capitalize()}{r.parameter.replace('_', '').replace('0', 'zero').replace('tilde', 'tl')}"
        m(f"{key}Lo", f"{100 * r.frac_within_5pct_lower:.0f}")
        m(f"{key}Hi", f"{100 * r.frac_within_5pct_upper:.0f}")
        m(f"{key}Med", f"{r['median']:.3g}")


def param_correlations(df):
    out = []
    for model, names in (("SDPM", ["v0", "T", "s0", "a", "b", "s0_y", "s0_y_tilde", "lambda", "tau_lat"]),
                         ("RAU", ["Wcc", "eta", "xi", "tau", "v_pref", "Wb"])):
        ranks = df[[f"{model}_{n}" for n in names]].rank()
        c = ranks.corr()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out.append({"model": model, "param_a": a, "param_b": b,
                            "spearman_rho": float(c.loc[f"{model}_{a}", f"{model}_{b}"])})
    pc = pd.DataFrame(out)
    pc.to_csv(STAT / "parameter_correlations.csv", index=False)
    for model in ("SDPM", "RAU"):
        sub = pc[pc.model == model].assign(a=lambda x: x.spearman_rho.abs()).sort_values("a", ascending=False)
        top = sub.iloc[0]
        m(f"{model.capitalize()}TopPair", f"{SYMBOL_TEX[top.param_a]}--{SYMBOL_TEX[top.param_b]}")
        m(f"{model.capitalize()}TopRho", signed(top.spearman_rho, 2))
        m(f"{model.capitalize()}NStrongPairs", str(int((sub.a >= 0.5).sum())))
    rau = pc[pc.model == "RAU"].set_index(["param_a", "param_b"]).spearman_rho
    m("RhoEtaXi", signed(float(rau.loc[("eta", "xi")]), 2))
    m("RhoXiVpref", signed(float(rau.loc[("xi", "v_pref")]), 2))


def transfer(df):
    t = pd.read_csv(RES / "transferability_results.csv")
    lines = []
    for test_run, sub in t.groupby("test_run"):
        cells = [f"{int(sub.train_run.iloc[0])} $\\rightarrow$ {int(test_run)}", str(len(sub))]
        for model in ("SDPM", "RAU"):
            own = sub[f"{model}_calibrated_ADE"].to_numpy()
            tr = sub[f"{model}_transfer_ADE"].to_numpy()
            d = tr - own
            lo, hi = _boot_ci(d, np.median)
            cells += [f(np.median(own)), f(np.median(tr)), f"{signed(np.median(d))} {ci(lo, hi)}"]
            m(f"Tr{model.capitalize()}Own{RUN_WORD[int(test_run)]}", f(np.median(own)))
            m(f"Tr{model.capitalize()}Tr{RUN_WORD[int(test_run)]}", f(np.median(tr)))
            m(f"Tr{model.capitalize()}D{RUN_WORD[int(test_run)]}", signed(np.median(d)))
            m(f"Tr{model.capitalize()}Better{RUN_WORD[int(test_run)]}", str(int((d < 0).sum())))
        lines.append(" & ".join(cells) + " \\\\")
    for model in ("SDPM", "RAU"):
        d = (t[f"{model}_transfer_ADE"] - t[f"{model}_calibrated_ADE"]).to_numpy()
        lo, hi = _boot_ci(d, np.median)
        m(f"Tr{model.capitalize()}Dall", signed(np.median(d)))
        m(f"Tr{model.capitalize()}Dlo", signed(lo))
        m(f"Tr{model.capitalize()}Dhi", signed(hi))
        m(f"Tr{model.capitalize()}Betterall", str(int((d < 0).sum())))
    write("tab_transfer", "\n".join(lines) + "\n")


def knockout():
    k = pd.read_csv(RES / "rau_ablation.csv")
    base = k.full_ADE.to_numpy()
    rows = [("Calibrated (all terms)", "full_ADE", "full_RMSE_d")]
    rows += [("$W_{cc}=0$ (no conflict term)", "Wcc0_ADE", "Wcc0_RMSE_d"),
             ("$W_b=0$ (no corridor term)", "Wb0_ADE", "Wb0_RMSE_d"),
             ("$T_{\\mathrm{ant}}=1.6$ s", "Tant_1.6_ADE", None),
             ("$T_{\\mathrm{ant}}=2.4$ s", "Tant_2.4_ADE", None)]
    lines = []
    tags = {"Wcc0_ADE": "Wcc", "Wb0_ADE": "Wb", "Tant_1.6_ADE": "Tlow", "Tant_2.4_ADE": "Thigh"}
    for label, col, dcol in rows:
        v = k[col].to_numpy()
        rmse_d = f(np.median(k[dcol]), 2) if dcol else "--"
        if col == "full_ADE":
            lines.append(f"{label} & {f(np.median(v))} & -- & {rmse_d} & -- \\\\")
            m("KoFull", f(np.median(v)))
            continue
        d = v - base
        lo, hi = _boot_ci(d, np.median)
        worse = int((d > 1e-9).sum())
        lines.append(f"{label} & {f(np.median(v))} & {signed(np.median(d), 2)} {ci(lo, hi, 2)} & {rmse_d} & {worse} \\\\")
        tag = tags[col]
        m(f"Ko{tag}", f(np.median(v)))
        m(f"Ko{tag}D", signed(np.median(d), 2))
        m(f"Ko{tag}Dlo", signed(lo, 2))
        m(f"Ko{tag}Dhi", signed(hi, 2))
        m(f"Ko{tag}Worse", str(worse))
        m(f"Ko{tag}Same", str(int((np.abs(d) <= 1e-9).sum())))
        m(f"Ko{tag}MeanD", signed(float(np.mean(d)), 2))
    m("KoWccRmsed", f(np.median(k.Wcc0_RMSE_d), 2))
    m("KoFullRmsed", f(np.median(k.full_RMSE_d), 2))
    m("KoWccRmsedD", signed(float(np.median(k.Wcc0_RMSE_d - k.full_RMSE_d)), 2))
    m("KoWbBetter", str(int((k.Wb0_ADE - k.full_ADE < -1e-9).sum())))
    write("tab_knockout", "\n".join(lines) + "\n")


def replicates():
    r = pd.read_csv(STAT / "noise_replicate_tests.csv").set_index("metric")
    a = r.loc["ADE"]
    m("RepN", str(int(a.replicates)))
    m("RepSdpmADE", f(a.sdpm_median))
    m("RepRauADE", f(a.rau_median))
    m("RepDmedADE", signed(a.delta_median))
    m("RepDloADE", signed(a.ci95_median_low))
    m("RepDhiADE", signed(a.ci95_median_high))
    m("RepPADE", p_fmt(a.p_wilcoxon))
    m("RepNrauADE", str(int(a.n_rau_better)))
    s = r.loc["RMSE_s"]
    m("RepDmedRMSEs", signed(s.delta_median))
    m("RepDloRMSEs", signed(s.ci95_median_low))
    m("RepDhiRMSEs", signed(s.ci95_median_high))
    d = r.loc["RMSE_d"]
    m("RepDmedRMSEd", signed(d.delta_median))
    m("RepDloRMSEd", signed(d.ci95_median_low))
    m("RepDhiRMSEd", signed(d.ci95_median_high))
    reps = pd.read_csv(STAT / "noise_replicates.csv")
    sd = reps.groupby(["canonical_id", "model"]).ADE.std().unstack("model").median()
    m("RepSdSdpm", f(sd["SDPM"]))
    m("RepSdRau", f(sd["RAU"]))


def regime():
    rc = pd.read_csv(STAT / "regime_correlations.csv")
    key = rc[(rc.descriptor == "frac_forward_beyond_1_5m") & (rc.delta == "delta_ADE")].iloc[0]
    m("RegRho", signed(key.spearman_rho, 2))
    m("RegRholo", signed(key.rho_ci95_low, 2))
    m("RegRhohi", signed(key.rho_ci95_high, 2))
    m("RegMaxAbsRho", f(rc.spearman_rho.abs().max(), 2))
    stag = rc[rc.descriptor.isin(["median_forward_lateral_m", "p90_forward_lateral_m", "frac_forward_beyond_1_5m"])]
    m("RegStagMaxAbsRho", f(stag.spearman_rho.abs().max(), 2))
    m("RegStagNexcl", str(int(((stag.rho_ci95_low > 0) | (stag.rho_ci95_high < 0)).sum())))
    lat = rc[(rc.descriptor == "lat_pos_std_m") & (rc.delta == "delta_ADE")].iloc[0]
    m("RegLatRho", signed(lat.spearman_rho, 2))
    m("RegLatRholo", signed(lat.rho_ci95_low, 2))
    m("RegLatRhohi", signed(lat.rho_ci95_high, 2))
    lats = rc[(rc.descriptor == "lat_pos_std_m") & (rc.delta == "delta_RMSE_s")].iloc[0]
    m("RegLatsRho", signed(lats.spearman_rho, 2))
    latd = rc[(rc.descriptor == "lat_pos_std_m") & (rc.delta == "delta_RMSE_d")].iloc[0]
    m("RegLatdRho", signed(latd.spearman_rho, 2))
    m("RegLatdRholo", signed(latd.rho_ci95_low, 2))
    m("RegLatdRhohi", signed(latd.rho_ci95_high, 2))
    n_excl = int(((rc.rho_ci95_low > 0) | (rc.rho_ci95_high < 0)).sum())
    m("RegNexclZero", str(n_excl))
    m("RegNcorr", str(len(rc)))


def cohort():
    c = pd.read_csv(ROOT / "outputs" / "cohorts" / "paired_cohort.csv")
    for split in ("pilot", "reserve", "main"):
        m(f"Cohort{split.capitalize()}", str(int((c.split == split).sum())))


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "paired_model_results.csv")
    steps = [data_table, stagger, cohort, lambda: comparison(df), lambda: streams(df), horizon,
             lambda: bounds(df), lambda: param_correlations(df), lambda: transfer(df), knockout,
             replicates, regime]
    missing = []
    for step in steps:
        try:
            step()
        except (FileNotFoundError, KeyError, AttributeError) as exc:
            missing.append(repr(exc))
    for msg in missing:
        print("SKIPPED:", msg)
    body = "% Generated by analysis/generate_tables.py. Do not edit by hand.\n"
    body += "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items()))
    (PAPER / "numbers.tex").write_text(body)
    for name in ("tab_paired", "tab_horizon", "tab_params"):
        if (PAPER_TABLES / f"{name}.tex").exists():
            shutil.copy(PAPER_TABLES / f"{name}.tex", TAB / f"{name}_rows.tex")
    print(f"{len(macros)} macros written")


if __name__ == "__main__":
    main()
