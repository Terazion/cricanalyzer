"""Phase 4: per-delivery player value in runs, plus win probability added (WPA)."""
from __future__ import annotations

import numpy as np
import pandas as pd

FLAG_COLS = ["pressure", "pressure_rule"]


def _flag_cols(df):
    return [c for c in df.columns if c == "pressure_rule" or c == "pressure" or c.startswith("pressure_")]


def compute_values(df: pd.DataFrame, cfg: dict) -> dict:
    d = df[df["scored"]].copy()
    v = cfg["value"]

    # runs <-> win probability exchange rate: average WP gained per run across all scored states
    wp_per_run = float(((d["wp_if_six"] - d["wp_if_dot"]) / 6.0).mean())
    wp_per_run = max(wp_per_run, 1e-4)
    if v.get("wicket_cost_conversion", "local") == "run_expectancy":
        raw_cost = pd.Series(_run_expectancy_cost(df, d, cfg), index=d.index)
    elif v.get("wicket_cost_conversion", "local") == "local":
        local = ((d["wp_if_six"] - d["wp_if_dot"]) / 6.0).clip(lower=wp_per_run * v.get("wp_per_run_floor_share", 0.1))
        raw_cost = d["wicket_wp_cost"] / local
    else:
        raw_cost = d["wicket_wp_cost"] / wp_per_run
    d["wicket_cost_runs"] = np.clip(raw_cost, v["wicket_cost_floor_runs"], v["wicket_cost_cap_runs"])

    # WPA: change in batting-side WP across the delivery
    g = d.groupby(["match_id", "innings"], sort=False)
    nxt = g["wp_before"].shift(-1)
    first_inn2 = d[d["innings"] == 2].groupby("match_id")["wp_before"].first()
    end1 = 1 - d["match_id"].map(first_inn2)
    end = np.where(d["innings"] == 1, end1.fillna(d["bat_win"]), d["bat_win"])
    d["wpa"] = (nxt.fillna(pd.Series(end, index=d.index)) - d["wp_before"]).fillna(0.0)

    faced = d["faced"].to_numpy().astype(float)
    cost = d["wicket_cost_runs"].to_numpy()
    striker_out = (d["out_player_id"] == d["striker_id"]).to_numpy()
    other_out = d["out_player_id"].notna().to_numpy() & ~striker_out
    d["bat_value"] = (faced * (d["runs"] - d["xr_bat"]) + d["p_dismissal"] * cost - striker_out * cost)
    d["bowl_value"] = (d["xr_bowl"] - d["bowler_runs"]
                       + (d["bowler_wicket"].astype(float) - d["p_bowler_wicket"]) * cost)

    flags = _flag_cols(d)
    common = ["match_id", "season", "comp_type", "match_dt", "innings"] + flags
    bat = d[common].copy()
    bat["player_id"] = d["striker_id"].to_numpy()
    bat["team_id"], bat["opp_id"] = d["batting_team_id"].to_numpy(), d["bowling_team_id"].to_numpy()
    bat["balls"] = faced
    bat["value"] = d["bat_value"].to_numpy()
    bat["wpa"] = np.where(other_out, 0.0, d["wpa"])
    bat["runs"] = d["runs"].to_numpy()
    bat["outs"] = striker_out.astype(int)
    bat["fours"], bat["sixes"] = (d["runs"] == 4).astype(int).to_numpy(), (d["runs"] == 6).astype(int).to_numpy()
    # non-striker (or other) dismissals: the cost goes to dismissed_player_id, with no ball faced
    o = d[other_out]
    extra = o[common].copy()
    extra["player_id"] = o["out_player_id"].to_numpy()
    extra["team_id"], extra["opp_id"] = o["batting_team_id"].to_numpy(), o["bowling_team_id"].to_numpy()
    extra["balls"], extra["value"], extra["wpa"] = 0.0, -o["wicket_cost_runs"].to_numpy(), o["wpa"].to_numpy()
    extra["runs"], extra["outs"], extra["fours"], extra["sixes"] = 0, 1, 0, 0
    bat = pd.concat([bat, extra], ignore_index=True)
    bat["player_id"] = bat["player_id"].astype("int64")

    bowl = d[common].copy()
    bowl["player_id"] = d["bowler_id"].to_numpy()
    bowl["team_id"], bowl["opp_id"] = d["bowling_team_id"].to_numpy(), d["batting_team_id"].to_numpy()
    bowl["balls"] = d["is_legal"].to_numpy().astype(float)
    bowl["value"] = d["bowl_value"].to_numpy()
    bowl["wpa"] = -d["wpa"].to_numpy()
    bowl["runs"] = d["bowler_runs"].to_numpy()
    bowl["wickets"] = d["bowler_wicket"].astype(int).to_numpy()
    bowl["dots"] = ((d["bowler_runs"] == 0) & d["is_legal"]).astype(int).to_numpy()
    bowl["boundaries"] = ((d["runs"] == 4) | (d["runs"] == 6)).astype(int).to_numpy()
    bowl["player_id"] = bowl["player_id"].astype("int64")

    # diagnostics: is the baseline centred inside each segment, and what drives the noise?
    diag = []
    run_part = faced * (d["runs"] - d["xr_bat"]).to_numpy()
    wkt_part = d["bat_value"].to_numpy() - run_part
    brun_part = (d["xr_bowl"] - d["bowler_runs"]).to_numpy()
    bwkt_part = d["bowl_value"].to_numpy() - brun_part
    for seg, m in [("pressure", d["pressure"].to_numpy()), ("non-pressure", ~d["pressure"].to_numpy())]:
        diag.append({
            "segment": seg, "deliveries": int(m.sum()),
            "bat_mean_value_per_ball": d["bat_value"].to_numpy()[m].sum() / max(faced[m].sum(), 1),
            "bat_sd_runs_part": run_part[m].std(), "bat_sd_wicket_part": wkt_part[m].std(),
            "bat_wicket_share_of_variance": wkt_part[m].var() / max(d["bat_value"].to_numpy()[m].var(), 1e-12),
            "bowl_mean_value_per_ball": d["bowl_value"].to_numpy()[m].mean(),
            "bowl_sd_runs_part": brun_part[m].std(), "bowl_sd_wicket_part": bwkt_part[m].std(),
            "bowl_wicket_share_of_variance": bwkt_part[m].var() / max(d["bowl_value"].to_numpy()[m].var(), 1e-12),
            "mean_wicket_cost_runs": d["wicket_cost_runs"].to_numpy()[m].mean(),
            "actual_dismissal_rate": d["is_wicket"].to_numpy()[m].mean(),
            "expected_dismissal_rate": d["p_dismissal"].to_numpy()[m].mean(),
            "actual_bat_runs_per_ball": d["runs"].to_numpy()[m].sum() / max(faced[m].sum(), 1),
            "expected_bat_runs_per_ball": (faced * d["xr_bat"].to_numpy())[m].sum() / max(faced[m].sum(), 1)})
    diag = pd.DataFrame(diag)
    cost_q = d["wicket_cost_runs"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2).to_dict()

    return {"df": d, "bat": bat, "bowl": bowl, "wp_per_run": wp_per_run, "diagnostics": diag, "cost_quantiles": cost_q,
            "conversion": v.get("wicket_cost_conversion", "local"),
            "mean_wicket_cost": float(d["wicket_cost_runs"].mean()),
            "mean_wicket_cost_pressure": float(d.loc[d["pressure"], "wicket_cost_runs"].mean()),
            "share_cost_capped": float((d["wicket_cost_runs"] >= v["wicket_cost_cap_runs"]).mean())}


def _run_expectancy_cost(df_all, d, cfg):
    """Wicket cost = E[remaining runs | balls, wickets] - E[remaining runs | balls, wickets + 1].

    Fitted on first innings of all matches (chases stop early), monotone in balls and wickets.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    t = df_all[(df_all["innings"] == 1) & df_all["state_valid"]]
    rem = (t["inn1_total"] - t["score_before"]).to_numpy(dtype=float)
    X = t[["balls_remaining", "wickets_before"]].to_numpy(dtype=float)
    m = HistGradientBoostingRegressor(monotonic_cst=[1, -1], max_iter=200, learning_rate=0.06, max_depth=4,
                                      min_samples_leaf=200, random_state=cfg["seed"]).fit(X, rem)
    b = np.maximum(d["balls_remaining"].to_numpy(dtype=float) - 1, 0)
    w = d["wickets_before"].to_numpy(dtype=float)
    return np.maximum(m.predict(np.column_stack([b, w])) - m.predict(np.column_stack([b, np.minimum(w + 1, 10)])), 0)


def player_match(rec: pd.DataFrame, flag: str = "pressure", value_col: str = "adj_value") -> pd.DataFrame:
    """One row per player per match with pressure / non-pressure sums."""
    p = rec[flag].to_numpy().astype(float)
    t = pd.DataFrame({
        "player_id": rec["player_id"].to_numpy(), "match_id": rec["match_id"].to_numpy(),
        "bp": rec["balls"].to_numpy() * p, "vp": rec[value_col].to_numpy() * p,
        "bnp": rec["balls"].to_numpy() * (1 - p), "vnp": rec[value_col].to_numpy() * (1 - p),
        "wpa_p": rec["wpa"].to_numpy() * p, "raw_vp": rec["value"].to_numpy() * p,
    })
    pm = t.groupby(["player_id", "match_id"], as_index=False).sum()
    meta = rec.drop_duplicates("match_id").set_index("match_id")[["season", "comp_type", "match_dt"]]
    return pm.join(meta, on="match_id")
