"""Synthetic data generator with the audited schema. FOR TESTS ONLY.

Used to exercise the pipeline when the real data/ folder is unavailable. Never
write its output into data/ or outputs/: rankings must come from the real data.

    python tests/synth.py OUT_DIR [N_MATCHES]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

WEAK = {"Hong Kong", "Kuwait", "Singapore", "Japan", "Nepal", "Oman"}


def generate(out_dir: str, n_matches: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    comps, teams, mapping, players = [], [], [], {}
    pid = [1000]

    def squad(team_id, strength):
        out = []
        for i in range(13):
            pid[0] += 1
            players[pid[0]] = {"name": f"Player {pid[0]}", "bat": rng.normal(strength, 0.12) - 0.02 * i,
                               "bowl": rng.normal(strength, 0.12), "bowler": i >= 6,
                               "type": rng.choice(["pace", "spin"])}
            out.append(pid[0])
        return out

    ipl_teams = {i: f"IPL Team {i}" for i in range(1, 9)}
    nat = ["India", "Australia", "England", "Pakistan", "New Zealand", "South Africa", "Hong Kong", "Kuwait",
           "Singapore", "Japan", "Nepal", "Oman"]
    nat_teams = {100 + i: n for i, n in enumerate(nat)}
    a_teams = {200 + i: f"A Team {i}" for i in range(4)}
    squads = {t: squad(t, 0.0) for t in ipl_teams}
    squads.update({t: squad(t, -0.35 if n in WEAK else 0.05) for t, n in nat_teams.items()})
    squads.update({t: squad(t, -0.1) for t in a_teams})
    # one name shared by two ids, one id with two spellings
    ids = list(players)
    players[ids[5]]["name"] = players[ids[40]]["name"] = "Shared Name"
    cid = 10
    comp_teams = {}
    for yr in range(2021, 2025):
        cid += 1
        comps.append((cid, f"IPL {yr}", "T20", str(yr))); comp_teams[cid] = ipl_teams
        cid += 1
        comps.append((cid, f"T20I Series {yr}", "T20I", f"{yr}/{str(yr + 1)[2:]}")); comp_teams[cid] = nat_teams
    for c in (2670, 3289):
        comps.append((c, f"A Team Tri-Series {c}", "T20", "2023")); comp_teams[c] = a_teams
    for c, tm in comp_teams.items():
        cname = [x[1] for x in comps if x[0] == c][0]
        for t, n in tm.items():
            teams.append((c, t, n))
            for p in squads[t]:
                mapping.append((c, cname, t, n, p, players[p]["name"]))
    mapping.append((11, "IPL 2021", 1, ipl_teams[1], ids[0], players[ids[0]]["name"] + " "))  # spelling variant

    balls, matches, xi = [], [], []
    comp_ids = [c[0] for c in comps]
    probs = np.array([0.36, 0.36] * 4 + [0.03, 0.03]); probs = probs / probs.sum()

    def sim_innings(mid, inn, bat_t, bowl_t, target, length, max_overs=20):
        bats = squads[bat_t][:11]
        bowlers = [p for p in squads[bowl_t][:11] if players[p]["bowler"]]
        s, ns, nxt = 0, 1, 2
        score = wk = legal = n = 0
        obn = 0
        while legal < length and wk < 10:
            over = legal // 6 + 1
            if legal % 6 == 0 and obn != 0 or n == 0:
                obn = 0
                bowler = bowlers[(over * 7 + mid) % len(bowlers)]
            n += 1; obn += 1
            b, w = players[bats[s]], players[bowler]
            edge = b["bat"] - w["bowl"]
            rem = length - legal
            agg = 0.25 if over > 15 else (0.1 if over <= 6 else 0.0)
            if target is not None:
                rrr = (target - score) * 6 / rem
                agg += np.clip((rrr - 8) / 20, -0.1, 0.4)
            agg -= 0.03 * wk
            p = np.array([0.36 - 0.1 * agg - 0.05 * edge, 0.36, 0.07, 0.005, 0.11 + 0.06 * agg + 0.05 * edge,
                          0.05 + 0.07 * agg + 0.04 * edge, 0.045 + 0.03 * agg - 0.02 * edge])
            p = np.clip(p, 0.002, None); p /= p.sum()
            runs = [0, 1, 2, 3, 4, 6, 0][(k := rng.choice(7, p=p))]
            is_w = k == 6
            row = dict(match_id=mid, innings=inn, overid=over, over_ball_num=obn, innings_ball_num=n,
                       batting_team_id=bat_t, bowling_team_id=bowl_t, striker_id=bats[s],
                       striker_name=b["name"], non_striker_id=bats[ns], non_striker_name=players[bats[ns]]["name"],
                       bowler_id=bowler, bowler_name=w["name"], batting_type="RHB", bowling_type="x",
                       batting_order=s + 1, runs=0, is_four=False, is_six=False, is_wicket=False,
                       dismissal_type_id=np.nan, dismissal_type=np.nan, dismissed_player_id=np.nan,
                       extras_type=np.nan, extras1_runs=0, extras_2_runs=0, pace_or_spin=w["type"], bowl_speed=0,
                       stroke="", delivery_style="", over_or_around="", on_or_off="", delivery_ist_time="")
            u = rng.random()
            is_legal = True
            if u < 0.03:
                row.update(extras_type="wide", extras1_runs=1, extras_2_runs=int(rng.random() < 0.05) * 4)
                is_legal, runs, is_w = False, 0, False
            elif u < 0.036:
                row.update(extras_type=rng.choice(["Overstep", "other"]), extras1_runs=1, runs=runs)
                is_legal, is_w = False, False
            elif u < 0.05:
                row.update(extras_type="legbyes", extras1_runs=1); runs = 0; is_w = False
            elif u < 0.053:
                row.update(extras_type="byes", extras1_runs=int(rng.choice([1, 4]))); runs = 0; is_w = False
            elif u < 0.0535:
                row.update(extras_type="Penalty", extras_2_runs=5, runs=runs); is_w = False
            else:
                row["runs"] = runs
            if is_w:
                kind = rng.choice(["Caught", "Bowled", "LBW", "Stump", "Hit Wicket", "RunOut", "Retire Out",
                                   "Obstracting the field"], p=[0.55, 0.17, 0.1, 0.04, 0.01, 0.11, 0.01, 0.01])
                out_ns = kind == "RunOut" and rng.random() < 0.45
                row.update(is_wicket=True, dismissal_type=kind, dismissed_player_id=bats[ns] if out_ns else bats[s], runs=0)
                wk += 1
                if nxt < 11:
                    if out_ns: ns = nxt
                    else: s = nxt
                    nxt += 1
            elif rng.random() < 0.0001:
                row["dismissed_player_id"] = bats[s]  # id present, not a wicket
            row["is_four"], row["is_six"] = row["runs"] == 4, row["runs"] == 6
            tr = row["runs"] + row["extras1_runs"] + row["extras_2_runs"]
            score += tr
            legal += is_legal
            balls.append(row)
            if (row["runs"] + (row["extras1_runs"] if row["extras_type"] in ("legbyes", "byes") else 0)) % 2 == 1:
                s, ns = ns, s
            if is_legal and legal % 6 == 0:
                s, ns = ns, s
            if target is not None and score >= target:
                break
        return score, wk, legal

    for i in range(n_matches):
        mid = 5000 + i
        c = comp_ids[i % len(comp_ids)]
        tm = comp_teams[c]
        h, a = rng.choice(list(tm), 2, replace=False)
        season = [x[3] for x in comps if x[0] == c][0][:4]
        m = dict(match_id=mid, comp_id=c, match_number=i, match_date=f"{season}-0{1 + i % 9}-{10 + i % 18}", stage="League",
                 venue_id=int(rng.integers(1, 20)), venue_country="X", home_team_id=h, home_team=tm[h], away_team_id=a,
                 away_team=tm[a], home_captain_id=squads[h][0], away_captain_id=squads[a][0], toss_win_team_id=h,
                 toss_opted="bat", man_of_the_match=np.nan, first_inn_revised_overs=np.nan,
                 second_inn_revised_overs=np.nan, revised_target=np.nan, knock_out=int(i % 17 == 0), win_team_id=np.nan,
                 result="")
        if m["knock_out"]:
            m["stage"] = "Final"
        if i % 53 == 7:  # abandoned: no ball data
            m["result"] = "Match abandoned"; matches.append(m); continue
        for t in (h, a):
            for j, p in enumerate(squads[t]):
                xi.append((mid, t, p, j + 1, j < 11, j == 0, j == 3, False, False))
        t1, _, _ = sim_innings(mid, 1, h, a, None, 120)
        kind = "dls_rev" if i % 41 == 3 else ("dls_norev" if i % 43 == 5 else ("nr" if i % 47 == 9 else "normal"))
        if kind == "nr":
            m["result"] = "No result"; matches.append(m); continue
        length, target = 120, t1 + 1
        if kind == "dls_rev":
            length, target = 90, int(t1 * 0.8) + 1
            m.update(second_inn_revised_overs=90, revised_target=target)
        elif kind == "dls_norev":
            length, target = 72, int(t1 * 0.65) + 1
        t2, w2, _ = sim_innings(mid, 2, a, h, target, length)
        suffix = " (D/L method)" if kind == "dls_rev" else (" (DLS)" if kind == "dls_norev" else "")
        if t2 >= target:
            m.update(win_team_id=a, result=f"{tm[a]} Won by {10 - w2} wickets{suffix}")
        elif t2 == target - 1:
            for inn, (bt, bw) in zip((3, 4), ((a, h), (h, a))):
                sim_innings(mid, inn, bt, bw, None, 6)
            m.update(win_team_id=h, result=f"Match tied ({tm[h]} won the Super Over)")
        else:
            m.update(win_team_id=h, result=f"{tm[h]} Won by {target - 1 - t2} runs{suffix}")
        matches.append(m)

    pd.DataFrame(balls).to_csv(os.path.join(out_dir, "ball_by_ball.csv"), index=False)
    pd.DataFrame(matches).to_csv(os.path.join(out_dir, "matches.csv"), index=False)
    pd.DataFrame(comps, columns=["comp_id", "comp_name", "match_format", "season"]).to_csv(os.path.join(out_dir, "competitions.csv"), index=False)
    pd.DataFrame(teams, columns=["comp_id", "team_id", "team_name"]).to_csv(os.path.join(out_dir, "teams.csv"), index=False)
    pd.DataFrame(xi, columns=["match_id", "team_id", "player_id", "batting_order", "is_played", "is_captain", "is_wk",
                              "is_impact", "is_impact_off"]).to_csv(os.path.join(out_dir, "playing_eleven.csv"), index=False)
    pd.DataFrame(mapping, columns=["comp_id", "comp_name", "team_id", "team_name", "player_id", "player_name"]).to_csv(
        os.path.join(out_dir, "competition_team_player_mapping.csv"), index=False)
    pd.DataFrame([(k, v["bat"], v["bowl"]) for k, v in players.items()], columns=["player_id", "bat_skill", "bowl_skill"]).to_csv(
        os.path.join(out_dir, "_truth.csv"), index=False)
    return out_dir


if __name__ == "__main__":
    generate(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
