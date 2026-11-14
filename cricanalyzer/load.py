"""Loading, cleaning and the verified delivery-level data rules."""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import yaml

REQUIRED_FILES = [
    "ball_by_ball.csv", "matches.csv", "competitions.csv", "teams.csv",
    "playing_eleven.csv", "competition_team_player_mapping.csv",
]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _to_bool(s: pd.Series) -> pd.Series:
    """Robust boolean parse for True/False, 'True'/'False', 1/0, NaN."""
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) > 0
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "y", "t"])


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _season_year(season: pd.Series, match_date: pd.Series) -> pd.Series:
    yr = season.astype(str).str.extract(r"((?:19|20)\d{2})")[0].astype(float)
    dt = pd.to_datetime(match_date, errors="coerce", format="mixed")
    return yr.fillna(dt.dt.year)


def load_data(cfg: dict) -> dict:
    d = cfg["paths"]["data_dir"]
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(d, f))]
    if missing:
        raise FileNotFoundError(f"Missing data files in '{d}': {missing}")

    bbb = pd.read_csv(os.path.join(d, "ball_by_ball.csv"), low_memory=False)
    matches = pd.read_csv(os.path.join(d, "matches.csv"), low_memory=False)
    comps = pd.read_csv(os.path.join(d, "competitions.csv"))
    teams = pd.read_csv(os.path.join(d, "teams.csv"))
    mapping = pd.read_csv(os.path.join(d, "competition_team_player_mapping.csv"))
    n_raw_rows, n_raw_matches = len(bbb), bbb["match_id"].nunique()

    # ---------------- competitions / matches
    comps = comps.drop_duplicates("comp_id").copy()
    name = comps["comp_name"].astype(str).str.strip()
    fmt = comps["match_format"].astype(str).str.strip().str.upper()
    comps["comp_type"] = np.where(name.str.upper().str.startswith("IPL"), "IPL",
                                  np.where(fmt == "T20I", "T20I", "OTHER"))
    matches = matches.drop_duplicates("match_id").merge(
        comps[["comp_id", "comp_name", "comp_type", "season"]], on="comp_id", how="left")
    matches["comp_type"] = matches["comp_type"].fillna("OTHER")
    matches["season_year"] = _season_year(matches["season"], matches["match_date"])
    matches["match_dt"] = pd.to_datetime(matches["match_date"], errors="coerce", format="mixed")
    matches["knock_out"] = _to_bool(matches["knock_out"]).astype(int)
    for c in ["first_inn_revised_overs", "second_inn_revised_overs", "revised_target", "win_team_id"]:
        matches[c] = _num(matches[c])
        if c != "win_team_id":
            matches.loc[matches[c] <= 0, c] = np.nan
    res = matches["result"].fillna("").astype(str)
    matches["is_dls"] = res.str.contains(r"d/l|dls", case=False, regex=True)
    matches["result_says_tie"] = res.str.contains(r"\btie|tied|super over", case=False, regex=True)

    # ---------------- ball by ball
    has_super_over = bbb.loc[bbb["innings"] > 2, "match_id"].unique()
    matches["has_super_over"] = matches["match_id"].isin(has_super_over)
    bbb = bbb.copy()   # derived columns are computed for all innings; super overs are split off below
    for c in ["runs", "extras1_runs", "extras_2_runs"]:
        bbb[c] = _num(bbb[c]).fillna(0).astype(int)
    bbb["is_wicket"] = _to_bool(bbb["is_wicket"])
    et = bbb["extras_type"].fillna("").astype(str).str.strip().str.lower()
    rules = cfg["rules"]
    bbb["is_wide"] = et.isin(rules["wide_types"])
    bbb["is_noball"] = et.isin(rules["noball_types"])
    bbb["is_legal"] = ~(bbb["is_wide"] | bbb["is_noball"])
    bbb["faced"] = ~bbb["is_wide"]
    bbb["team_runs"] = bbb["runs"] + bbb["extras1_runs"] + bbb["extras_2_runs"]
    bbb["bowler_runs"] = bbb["runs"] + np.where(bbb["is_wide"] | bbb["is_noball"], bbb["extras1_runs"], 0)
    dt = bbb["dismissal_type"].fillna("").astype(str).str.strip().str.lower()
    bbb["bowler_wicket"] = bbb["is_wicket"] & dt.isin(rules["bowler_wicket_types"])
    bbb["dismissed_player_id"] = _num(bbb["dismissed_player_id"])
    # dismissed batter only where is_wicket is True (11 rows carry an id without a wicket)
    bbb["out_player_id"] = np.where(bbb["is_wicket"], bbb["dismissed_player_id"], np.nan)
    # a wicket row with no dismissed id falls back to the striker
    fb = bbb["is_wicket"] & bbb["out_player_id"].isna()
    bbb.loc[fb, "out_player_id"] = bbb.loc[fb, "striker_id"]
    bbb = bbb.sort_values(["match_id", "innings", "innings_ball_num"], kind="mergesort").reset_index(drop=True)
    super_overs = bbb[bbb["innings"] > 2].reset_index(drop=True)
    bbb = bbb[bbb["innings"].isin([1, 2])].reset_index(drop=True)

    # matches without ball data (abandoned) are ignored
    n_matches_file = len(matches)
    matches = matches[matches["match_id"].isin(bbb["match_id"].unique())].reset_index(drop=True)
    bbb = bbb[bbb["match_id"].isin(matches["match_id"])].reset_index(drop=True)

    # ---------------- canonical player names (one per player_id) and teams
    mp = mapping.dropna(subset=["player_id"]).copy()
    mp["player_name"] = mp["player_name"].astype(str).str.strip()
    cnt = mp.groupby(["player_id", "player_name"]).size().reset_index(name="n")
    cnt = cnt.sort_values(["player_id", "n", "player_name"], ascending=[True, False, True])
    names = cnt.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()
    for idc, nc in [("striker_id", "striker_name"), ("bowler_id", "bowler_name"),
                    ("non_striker_id", "non_striker_name")]:
        fallback = bbb.groupby(idc)[nc].agg(lambda x: x.value_counts().index[0])
        for pid, nm in fallback.items():
            names.setdefault(pid, str(nm).strip())

    team_names = (teams.dropna(subset=["team_id"]).groupby("team_id")["team_name"]
                  .agg(lambda x: x.value_counts().index[0]).to_dict())
    for a, b in [("home_team_id", "home_team"), ("away_team_id", "away_team")]:
        for tid, nm in matches.dropna(subset=[a]).groupby(a)[b].first().items():
            team_names.setdefault(tid, nm)

    return {
        "bbb": bbb, "super_overs": super_overs, "matches": matches, "names": names, "team_names": team_names,
        "meta": {"raw_rows": n_raw_rows, "raw_matches": n_raw_matches,
                 "matches_file_rows": n_matches_file,
                 "matches_without_ball_data": n_matches_file - len(matches),
                 "rows_regulation": len(bbb), "matches": len(matches),
                 "name_collisions": _name_collisions(names)},
    }


def _name_collisions(names: dict) -> int:
    s = pd.Series(names)
    return int((s.groupby(s).size() > 1).sum())


def parse_run_margin(result: str):
    """'... Won by 12 runs' -> 12, else None."""
    m = re.search(r"by\s+(\d+)\s+runs?\b", str(result), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None
