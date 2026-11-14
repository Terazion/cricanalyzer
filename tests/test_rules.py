"""Unit tests for the verified data rules on a tiny hand-built dataset."""
import os

import numpy as np
import pandas as pd
import pytest

from cricanalyzer import load, state

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ball(mid, inn, n, **kw):
    row = dict(match_id=mid, innings=inn, overid=(n - 1) // 6 + 1, over_ball_num=n, innings_ball_num=n,
               batting_team_id=1 if inn % 2 else 2, bowling_team_id=2 if inn % 2 else 1, striker_id=10, striker_name="A",
               non_striker_id=11, non_striker_name="B", bowler_id=20, bowler_name="C", runs=0, is_wicket=False,
               dismissal_type=np.nan, dismissed_player_id=np.nan, extras_type=np.nan, extras1_runs=0, extras_2_runs=0)
    row.update(kw)
    return row


@pytest.fixture()
def tiny(tmp_path):
    b = [
        _ball(1, 1, 1, runs=4),
        _ball(1, 1, 2, extras_type="wide", extras1_runs=1, extras_2_runs=4),          # wide + 4 byes off it
        _ball(1, 1, 3, extras_type="Overstep", extras1_runs=1, runs=6),
        _ball(1, 1, 4, extras_type="other", extras1_runs=1),
        _ball(1, 1, 5, extras_type="legbyes", extras1_runs=2),
        _ball(1, 1, 6, extras_type="Penalty", extras_2_runs=5),
        _ball(1, 1, 7, is_wicket=True, dismissal_type="RunOut", dismissed_player_id=11),
        _ball(1, 1, 8, is_wicket=True, dismissal_type="Caught", dismissed_player_id=10),
        _ball(1, 1, 9, is_wicket=False, dismissed_player_id=10),                      # id without a wicket
        _ball(1, 1, 10, is_wicket=True, dismissal_type="Obstracting the field", dismissed_player_id=10),
        _ball(1, 2, 1, runs=1), _ball(1, 2, 2, runs=2),
        _ball(1, 3, 1, runs=6), _ball(1, 4, 1, runs=4),                               # super over
        _ball(2, 1, 1, runs=6), _ball(2, 2, 1, runs=1),
        _ball(3, 1, 1, runs=6), _ball(3, 2, 1, runs=1),
    ]
    pd.DataFrame(b).to_csv(tmp_path / "ball_by_ball.csv", index=False)
    m = lambda mid, comp, **kw: dict(dict(match_id=mid, comp_id=comp, match_date="2023-04-01", stage="League", venue_id=1,
                                          home_team_id=1, home_team="X", away_team_id=2, away_team="Y",
                                          first_inn_revised_overs=np.nan, second_inn_revised_overs=np.nan,
                                          revised_target=np.nan, knock_out=0, win_team_id=1, result="X Won by 3 runs"), **kw)
    pd.DataFrame([m(1, 1, result="Match tied (X won the Super Over)"),
                  m(2, 2, second_inn_revised_overs=114, revised_target=5, result="X won by 2 runs (D/L method)"),
                  m(3, 3, result="X won by 1 run (dls)"),
                  m(4, 1, result="Abandoned", win_team_id=np.nan)]).to_csv(tmp_path / "matches.csv", index=False)
    pd.DataFrame([(1, "IPL 2023", "T20", "2023"), (2, "Some Series", "T20I", "2022/23"),
                  (3, "A Tour", "T20", "2023")], columns=["comp_id", "comp_name", "match_format", "season"]).to_csv(
        tmp_path / "competitions.csv", index=False)
    pd.DataFrame([(1, 1, "X"), (1, 2, "Y")], columns=["comp_id", "team_id", "team_name"]).to_csv(tmp_path / "teams.csv", index=False)
    pd.DataFrame(columns=["match_id", "team_id", "player_id"]).to_csv(tmp_path / "playing_eleven.csv", index=False)
    pd.DataFrame([(1, "IPL 2023", 1, "X", 10, "Alpha"), (2, "S", 1, "X", 10, "Alpha"), (1, "IPL 2023", 1, "X", 10, "Alfa"),
                  (1, "IPL 2023", 1, "X", 30, "Alpha")],
                 columns=["comp_id", "comp_name", "team_id", "team_name", "player_id", "player_name"]).to_csv(
        tmp_path / "competition_team_player_mapping.csv", index=False)
    cfg = load.load_config(os.path.join(ROOT, "config.yaml"))
    cfg["paths"]["data_dir"] = str(tmp_path)
    data = load.load_data(cfg)
    return cfg, data, state.build_states(data["bbb"], data["matches"], cfg)


def test_load_rules(tiny):
    cfg, data, df = tiny
    b = data["bbb"]
    assert set(b["innings"]) == {1, 2}                      # super overs dropped
    assert set(data["matches"]["match_id"]) == {1, 2, 3}    # match without balls ignored
    x = b[(b.match_id == 1) & (b.innings == 1)].set_index("innings_ball_num")
    assert x["is_legal"].tolist() == [True, False, False, False, True, True, True, True, True, True]
    assert x["faced"].tolist() == [True, False] + [True] * 8
    assert x["team_runs"].tolist() == [4, 5, 7, 1, 2, 5, 0, 0, 0, 0]
    assert x["bowler_runs"].tolist() == [4, 1, 7, 1, 0, 0, 0, 0, 0, 0]
    assert x["bowler_wicket"].tolist() == [False] * 7 + [True, False, False]
    assert x.loc[7, "out_player_id"] == 11                  # run-out of the non-striker
    assert np.isnan(x.loc[9, "out_player_id"])              # dismissed id without is_wicket
    assert data["names"][10] == "Alpha"                     # one canonical name per id
    assert data["matches"].set_index("match_id")["comp_type"].to_dict() == {1: "IPL", 2: "T20I", 3: "OTHER"}


def test_state_rules(tiny):
    cfg, data, df = tiny
    x = df[(df.match_id == 1) & (df.innings == 1)].set_index("innings_ball_num")
    assert x["score_before"].tolist() == [0, 4, 9, 16, 17, 19, 24, 24, 24, 24]
    assert x["legal_before"].tolist() == [0, 1, 1, 1, 1, 2, 3, 4, 5, 6]
    assert x["wickets_before"].tolist() == [0] * 7 + [1, 2, 2]
    assert x.loc[10, "balls_remaining"] == 114
    c = df[(df.match_id == 1) & (df.innings == 2)]
    assert c["target"].iloc[0] == 25 and c["runs_needed"].tolist() == [25, 24]
    assert (df[df.match_id == 1]["bat_win"] == 0.5).all()   # super-over tie -> 0.5
    r = df[(df.match_id == 2) & (df.innings == 2)].iloc[0]  # revised target, overs given in balls
    assert r["target"] == 5 and r["balls_remaining"] == 114 and r["state_valid"] and r["wp_trainable"]
    d = df[df.match_id == 3]                                # DLS without revised target
    assert not d["wp_trainable"].any()
    assert d.loc[d.innings == 2, "state_valid"].eq(False).all() and d.loc[d.innings == 1, "state_valid"].all()


def test_super_over_margin_is_not_a_regulation_margin(tiny):
    cfg, data, df = tiny
    m = data["matches"].copy()
    m.loc[m.match_id == 1, "result"] = "X Won Super Over By 10 Runs"
    with pytest.raises(state.StateCheckError, match="regulation totals differ"):
        state.run_state_checks(df, m, cfg)          # tiny match 1 is 24 vs 3, so not level
    lvl = df[~((df.match_id == 1) & (df.innings == 1) & (df.innings_ball_num != 1))].copy()
    lvl.loc[(lvl.match_id == 1) & (lvl.innings == 1), "team_runs"] = 3
    state.run_state_checks(lvl, m, cfg)             # level totals pass despite "By 10 Runs"


def test_margin_check_catches_bad_totals(tiny):
    cfg, data, df = tiny
    m = data["matches"].copy()
    m.loc[m.match_id == 1, "result"] = "X Won by 999 runs"
    m.loc[m.match_id == 1, ["has_super_over", "result_says_tie"]] = False
    with pytest.raises(state.StateCheckError):
        state.run_state_checks(df, m, cfg)


def test_super_over_states(tiny):
    cfg, data, df = tiny
    so = state.build_super_over_states(data["super_overs"], data["matches"], cfg)
    assert len(so) == 2 and set(so["so_innings"]) == {3, 4}
    first, chase = so[so.so_innings == 3].iloc[0], so[so.so_innings == 4].iloc[0]
    assert first["innings"] == 1 and first["balls_remaining"] == 6 and np.isnan(first["rrr"])
    assert chase["innings"] == 2 and chase["target"] == 7 and chase["rrr"] == 7.0   # 6 scored, needs 7 off 6
    assert not (df["innings"] > 2).any()                                         # regulation frame untouched
