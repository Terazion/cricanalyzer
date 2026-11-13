"""Phase 1: match state BEFORE every delivery, plus hard integrity checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .load import parse_run_margin

COMP_CODE = {"IPL": 0, "T20I": 1, "OTHER": 2}


class StateCheckError(RuntimeError):
    pass


def build_states(bbb: pd.DataFrame, matches: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = bbb.copy()
    default = cfg["innings"]["default_balls"]
    g = df.groupby(["match_id", "innings"], sort=False)
    df["score_before"] = g["team_runs"].cumsum() - df["team_runs"]
    df["legal_before"] = g["is_legal"].cumsum() - df["is_legal"].astype(int)
    df["wickets_before"] = g["is_wicket"].cumsum() - df["is_wicket"].astype(int)

    tot = df.groupby(["match_id", "innings"])["team_runs"].sum().unstack()
    inn1_total = tot.get(1, pd.Series(dtype=float))
    inn2_total = tot.get(2, pd.Series(dtype=float))

    m = matches.set_index("match_id")
    cols = ["comp_id", "comp_type", "season_year", "venue_id", "knock_out", "stage", "match_dt",
            "first_inn_revised_overs", "second_inn_revised_overs", "revised_target",
            "is_dls", "win_team_id", "has_super_over", "result_says_tie"]
    df = df.join(m[cols], on="match_id")
    df["season"] = df["season_year"].fillna(df["season_year"].median()).astype(int)
    df["comp_code"] = df["comp_type"].map(COMP_CODE).astype(int)

    len1 = df["first_inn_revised_overs"].fillna(default)
    len2 = df["second_inn_revised_overs"].fillna(len1)  # shortened first innings carries over
    df["len2"] = len2.astype(int)
    df["innings_len"] = np.where(df["innings"] == 1, len1, len2).astype(int)
    # a delivery about to be bowled always has at least one ball remaining
    df["balls_remaining"] = np.maximum(df["innings_len"] - df["legal_before"], 1)
    pp, mid = cfg["innings"]["powerplay_end_over"], cfg["innings"]["middle_end_over"]
    df["phase"] = np.where(df["overid"] <= pp, "powerplay", np.where(df["overid"] <= mid, "middle", "death"))
    df["crr"] = np.where(df["legal_before"] > 0, df["score_before"] * 6 / df["legal_before"].clip(lower=1), 0.0)

    df["inn1_total"] = df["match_id"].map(inn1_total)
    target = df["revised_target"].fillna(df["inn1_total"] + 1)
    is2 = df["innings"] == 2
    df["target"] = np.where(is2, target, np.nan)
    df["runs_needed"] = np.where(is2, target - df["score_before"], np.nan)
    df["rrr"] = np.where(is2, df["runs_needed"] * 6 / df["balls_remaining"], np.nan)

    # DLS without a revised target: second-innings state is unknowable
    df["dls_unrevised"] = df["is_dls"] & df["revised_target"].isna()
    df["state_valid"] = ~(df["dls_unrevised"] & is2)

    # regulation result for the batting side: 1 win, 0 loss, 0.5 tie, NaN no result
    tie = df["has_super_over"] | df["result_says_tie"]
    lab = np.where(df["win_team_id"].isna(), np.nan,
                   np.where(tie, 0.5, (df["win_team_id"] == df["batting_team_id"]).astype(float)))
    df["bat_win"] = lab
    df["wp_trainable"] = ~np.isnan(lab) & ~df["dls_unrevised"]
    df["inn2_total"] = df["match_id"].map(inn2_total)
    return df


def run_state_checks(df: pd.DataFrame, matches: pd.DataFrame, cfg: dict) -> dict:
    """Raises StateCheckError listing every failed check."""
    fails = []
    legal = df.groupby(["match_id", "innings"])["is_legal"].sum()
    bad = legal[legal > cfg["innings"]["default_balls"]]
    if len(bad):
        fails.append(f"{len(bad)} innings exceed {cfg['innings']['default_balls']} legal balls: {bad.head(5).to_dict()}")
    wk = df.groupby(["match_id", "innings"])["is_wicket"].sum()
    bad = wk[wk > 10]
    if len(bad):
        fails.append(f"{len(bad)} innings have more than 10 wickets: {bad.head(5).to_dict()}")
    dup = df.duplicated(["match_id", "innings", "innings_ball_num"]).sum()
    if dup:
        fails.append(f"{dup} duplicate (match_id, innings, innings_ball_num) rows")

    tot = df.groupby(["match_id", "innings"])["team_runs"].sum().unstack()
    first_bat = df[df["innings"] == 1].groupby("match_id")["batting_team_id"].first()
    checked, mism = 0, []
    for r in matches.itertuples():
        margin = parse_run_margin(r.result)
        revised = any(pd.notna(getattr(r, c)) for c in
                      ["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"])
        if margin is None or r.is_dls or revised or r.match_id not in tot.index:
            continue
        t1, t2 = tot.loc[r.match_id].get(1, np.nan), tot.loc[r.match_id].get(2, np.nan)
        if np.isnan(t1) or np.isnan(t2):
            mism.append((r.match_id, "missing innings"))
            continue
        checked += 1
        if int(t1 - t2) != margin:
            mism.append((r.match_id, f"data margin {int(t1 - t2)} vs result '{r.result}'"))
        elif pd.notna(r.win_team_id) and first_bat.get(r.match_id) != r.win_team_id:
            mism.append((r.match_id, "winner did not bat first"))
    if mism:
        fails.append(f"{len(mism)} of {checked} 'won by N runs' matches do not reproduce: {mism[:8]}")
    if fails:
        raise StateCheckError("PHASE 1 CHECKS FAILED\n- " + "\n- ".join(fails))
    return {"run_margin_matches_checked": checked, "max_legal_balls": int(legal.max()),
            "max_wickets": int(wk.max())}
