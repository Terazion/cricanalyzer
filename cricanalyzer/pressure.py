"""Phase 3: leverage-based pressure index, rule-based cross-check, and evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import OUTCOME_LABELS, OUTCOME_RUNS


def compute_leverage(df: pd.DataFrame, wp_models: dict, cfg: dict) -> pd.DataFrame:
    """Adds wp_before, wp after each hypothetical outcome, leverage and wicket WP cost.

    Leverage = sum over outcomes o of P(o | state) * |WP(state after o) - WP(state)|.
    Each hypothetical outcome consumes one legal ball.
    """
    df = df.copy()
    n = len(df)
    wp_now = np.full(n, np.nan)
    wp_next = np.full((n, 6), np.nan)
    for k, model in wp_models.items():
        idx = np.where((df["fold"].to_numpy() == k) & df["state_valid"].to_numpy())[0]
        if len(idx) == 0:
            continue
        d = df.iloc[idx]
        inn, score, balls, wk = (d[c].to_numpy(dtype=float) for c in
                                 ["innings", "score_before", "balls_remaining", "wickets_before"])
        tgt, len2, comp, season = (d["target"].fillna(0).to_numpy(dtype=float), d["len2"].to_numpy(dtype=float),
                                   d["comp_code"].to_numpy(dtype=float), d["season"].to_numpy(dtype=float))
        wp_now[idx] = model.wp(inn, score, balls, wk, tgt, len2, comp, season)
        for j in range(6):
            wp_next[idx, j] = model.wp(inn, score + OUTCOME_RUNS[j], balls - 1, wk + (j == 5),
                                       tgt, len2, comp, season)
    po = df[[f"po_{l}" for l in OUTCOME_LABELS]].to_numpy()
    df["wp_before"] = wp_now
    df["wp_if_dot"], df["wp_if_six"], df["wp_if_wicket"] = wp_next[:, 0], wp_next[:, 4], wp_next[:, 5]
    df["leverage_raw"] = (po * np.abs(wp_next - wp_now[:, None])).sum(axis=1)
    mult = np.where(df["knock_out"].to_numpy() == 1, cfg["pressure"]["knockout_multiplier"], 1.0)
    df["leverage"] = df["leverage_raw"] * mult
    df["wicket_wp_cost"] = np.clip(df["wp_if_dot"] - df["wp_if_wicket"], 0, None)
    return df


def rule_based_flag(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    r = cfg["pressure"]["rule_based"]
    pp_end, mid_end = cfg["innings"]["powerplay_end_over"], cfg["innings"]["middle_end_over"]
    i2 = (df["innings"] == 2) & df["state_valid"]
    chase = i2 & (df["rrr"] >= r["chase_min_rrr"]) & (df["balls_remaining"] <= r["chase_max_balls_left"])
    collapse = (df["overid"] <= pp_end) & (df["wickets_before"] >= r["powerplay_min_wickets"])
    reach = i2 & (df["overid"] > mid_end) & df["rrr"].between(r["death_reach_min_rrr"], r["death_reach_max_rrr"])
    return (chase | collapse | reach).to_numpy()


def flag_pressure(df: pd.DataFrame, cfg: dict, rated_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Threshold = (1 - share) quantile of leverage over rated-competition deliveries."""
    df = df.copy()
    scored = rated_mask & df["leverage"].notna().to_numpy()
    lev = df.loc[scored, "leverage"]
    shares = sorted(set([cfg["pressure"]["top_share"]] + list(cfg["reliability"]["sensitivity_thresholds"])))
    thresholds = {}
    for s in shares:
        thr = float(lev.quantile(1 - s))
        thresholds[s] = thr
        df[f"pressure_{int(round(s * 100))}"] = (df["leverage"] >= thr).fillna(False)
    df["pressure"] = df[f"pressure_{int(round(cfg['pressure']['top_share'] * 100))}"]
    df["pressure_rule"] = rule_based_flag(df, cfg)
    df["scored"] = scored
    return df, thresholds


def evidence(df: pd.DataFrame) -> dict:
    """Tables showing the pressure definition behaves sensibly (rated, scored balls only)."""
    d = df[df["scored"]]
    by_over = d.groupby(["innings", "overid"])["pressure"].mean().unstack("innings")
    by_over.columns = [f"innings_{c}" for c in by_over.columns]

    def league(g):
        faced, legal = g["faced"].sum(), g["is_legal"].sum()
        return pd.Series({
            "deliveries": len(g),
            "strike_rate": 100 * g["runs"].sum() / max(faced, 1),
            "dismissal_rate_per_100_balls": 100 * g["is_wicket"].sum() / max(faced, 1),
            "economy": 6 * g["bowler_runs"].sum() / max(legal, 1),
            "boundary_pct": 100 * ((g["runs"] == 4) | (g["runs"] == 6)).sum() / max(faced, 1),
            "dot_pct": 100 * ((g["team_runs"] == 0) & g["is_legal"]).sum() / max(legal, 1),
            "mean_leverage": g["leverage"].mean(),
        })
    lab = np.where(d["pressure"], "pressure", "non-pressure")
    league_tab = pd.concat([league(g).rename(k) for k, g in d.groupby(lab)], axis=1).T
    by_phase = pd.concat([league(g).rename(k) for k, g in d.groupby([d["phase"], lab])], axis=1).T
    by_phase.index.names = ["phase", "segment"]

    cols = ["match_id", "comp_type", "season", "innings", "overid", "score_before", "wickets_before",
            "balls_remaining", "runs_needed", "rrr", "wp_before", "leverage", "knock_out"]
    one_per_match = d.sort_values("leverage", ascending=False).drop_duplicates("match_id")
    hi = one_per_match.head(8)[cols]
    lo = d.sort_values("leverage").drop_duplicates("match_id").head(8)[cols]
    q = d["leverage"].quantile([0.5, 0.75, 0.9, 0.99]).rename("leverage")

    overlap = pd.crosstab(d["pressure"], d["pressure_rule"], normalize="all")
    return {"by_over": by_over.reset_index(), "league": league_tab.reset_index(names="segment"),
            "by_phase": by_phase.reset_index(), "examples_high": hi, "examples_low": lo,
            "quantiles": q, "rule_overlap": overlap,
            "rule_share": float(d["pressure_rule"].mean()),
            "rule_agreement": float((d["pressure"] == d["pressure_rule"]).mean()),
            "share_inn": d.groupby("innings")["pressure"].mean(),
            "share_comp": d.groupby("comp_type")["pressure"].mean(),
            "share_ko": d.groupby("knock_out")["pressure"].mean()}
