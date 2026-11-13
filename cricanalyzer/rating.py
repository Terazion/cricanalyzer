"""Phase 5: opposition adjustment, empirical-Bayes shrinkage, composite rating."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .value import player_match


# ----------------------------------------------------------------- opposition strength
def opposition_strength(bat: pd.DataFrame, bowl: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Team-season batting and bowling strength in runs per ball (positive = strong).

    Solved iteratively so a team's strength is itself corrected for who it played:
      bat_str[t]  = sum(bat value  + balls * bowl_str[opp]) / (balls + prior)
      bowl_str[t] = sum(bowl value + balls * bat_str[opp])  / (balls + prior)
    """
    o = cfg["rating"]["opposition"]
    k = o["prior_balls"]
    agg = lambda r: r.groupby(["team_id", "season", "opp_id"], as_index=False)[["value", "balls"]].sum()
    a, b = agg(bat), agg(bowl)
    keys = pd.MultiIndex.from_frame(pd.concat([a[["team_id", "season"]], b[["team_id", "season"]],
                                               a[["opp_id", "season"]].rename(columns={"opp_id": "team_id"})])
                                    ).drop_duplicates()
    bat_str = pd.Series(0.0, index=keys)
    bowl_str = pd.Series(0.0, index=keys)
    a_t, a_o = pd.MultiIndex.from_frame(a[["team_id", "season"]]), pd.MultiIndex.from_frame(a[["opp_id", "season"]])
    b_t, b_o = pd.MultiIndex.from_frame(b[["team_id", "season"]]), pd.MultiIndex.from_frame(b[["opp_id", "season"]])
    for _ in range(o["iterations"]):
        num = pd.Series(a["value"].to_numpy() + a["balls"].to_numpy() * bowl_str.reindex(a_o).fillna(0).to_numpy(), index=a_t)
        den = pd.Series(a["balls"].to_numpy(), index=a_t)
        new_bat = (num.groupby(level=[0, 1]).sum() / (den.groupby(level=[0, 1]).sum() + k)).reindex(keys).fillna(0)
        num = pd.Series(b["value"].to_numpy() + b["balls"].to_numpy() * bat_str.reindex(b_o).fillna(0).to_numpy(), index=b_t)
        den = pd.Series(b["balls"].to_numpy(), index=b_t)
        new_bowl = (num.groupby(level=[0, 1]).sum() / (den.groupby(level=[0, 1]).sum() + k)).reindex(keys).fillna(0)
        bat_str, bowl_str = new_bat, new_bowl
    out = pd.DataFrame({"bat_strength": bat_str, "bowl_strength": bowl_str})
    out["bat_balls"] = a.groupby(["team_id", "season"])["balls"].sum().reindex(keys).fillna(0)
    out["bowl_balls"] = b.groupby(["team_id", "season"])["balls"].sum().reindex(keys).fillna(0)
    return out


def apply_opposition(bat, bowl, strengths, cfg):
    o = cfg["rating"]["opposition"]
    w = o["weight"] if o["enabled"] else 0.0
    for rec, col in [(bat, "bowl_strength"), (bowl, "bat_strength")]:
        idx = pd.MultiIndex.from_frame(rec[["opp_id", "season"]])
        s = strengths[col].reindex(idx).fillna(0).to_numpy()
        rec["opp_strength"] = s
        rec["adj_value"] = rec["value"] + w * rec["balls"] * s
    return bat, bowl


# ----------------------------------------------------------------- empirical Bayes
def fit_params(rec: pd.DataFrame, pm: pd.DataFrame, flag: str, cfg: dict) -> dict:
    """Population mean, per-ball variance and between-player variance, per segment."""
    r = cfg["rating"]
    prm = {}
    for seg, mask, bcol, vcol in [("p", rec[flag].to_numpy(), "bp", "vp"), ("np", ~rec[flag].to_numpy(), "bnp", "vnp")]:
        ball = rec[mask & (rec["balls"].to_numpy() > 0)]
        sig2 = float(ball["adj_value"].var()) if len(ball) > 1 else 1.0
        tot = pm.groupby("player_id")[[bcol, vcol]].sum()
        N = tot[bcol].sum()
        mu = float(tot[vcol].sum() / max(N, 1))
        t = tot[tot[bcol] >= r["eb_min_balls_for_variance"]]
        tau2 = 0.0
        if len(t) > 2:
            n, m = t[bcol].to_numpy(), (t[vcol] / t[bcol]).to_numpy()
            denom = n.sum() - (n ** 2).sum() / n.sum()
            tau2 = (np.sum(n * (m - mu) ** 2) - (len(t) - 1) * sig2) / max(denom, 1e-9)
        prm[f"mu_{seg}"], prm[f"sig2_{seg}"], prm[f"tau2_{seg}"] = mu, sig2, float(max(tau2, 1e-6))
        prm[f"tau2_{seg}_raw"] = float(tau2)
    # lift = pressure minus own non-pressure value per ball. The DIFFERENCE is shrunk directly
    # (DerSimonian-Laird); differencing two separately shrunk means would bias lift against
    # good players, because the smaller pressure sample is always shrunk harder.
    tot = pm.groupby("player_id")[["bp", "vp", "bnp", "vnp"]].sum()
    t = tot[(tot["bp"] >= r["eb_min_balls_for_variance"]) & (tot["bnp"] >= r["eb_min_balls_for_variance"])]
    prm["mu_d"], prm["tau2_d"] = prm["mu_p"] - prm["mu_np"], 1e-6
    if len(t) > 2:
        d = (t["vp"] / t["bp"] - t["vnp"] / t["bnp"]).to_numpy()
        wt = 1.0 / (prm["sig2_p"] / t["bp"] + prm["sig2_np"] / t["bnp"]).to_numpy()
        mu_d = float(np.sum(wt * d) / wt.sum())
        q = np.sum(wt * (d - mu_d) ** 2)
        prm["mu_d"] = mu_d
        prm["tau2_d_raw"] = float((q - (len(t) - 1)) / max(wt.sum() - (wt ** 2).sum() / wt.sum(), 1e-9))
        prm["tau2_d"] = float(max(prm["tau2_d_raw"], 1e-6))
    has = pm["bp"] >= r["consistency_min_balls"]
    prm["p0"] = float((pm.loc[has, "vp"] > 0).mean()) if has.any() else 0.5
    return prm


def _z(x, elig):
    out = np.full(len(x), np.nan)
    if elig.sum() > 1:
        sd = x[elig].std()
        out[elig] = (x[elig] - x[elig].mean()) / sd if sd > 0 else 0.0
    elif elig.sum() == 1:
        out[elig] = 0.0
    return out


def rate_arrays(pidx, n_players, bp, vp, bnp, vnp, w, prm, weights, elig, cmin, cprior):
    """Vectorised rating core; `w` are match weights (1 normally, bootstrap counts otherwise)."""
    bc = lambda x: np.bincount(pidx, weights=w * x, minlength=n_players)
    Np, Vp, Nn, Vn = bc(bp), bc(vp), bc(bnp), bc(vnp)

    def shrink(N, V, mu, sig2, tau2):
        m = np.where(N > 0, V / np.maximum(N, 1e-9), mu)
        B = np.where(N > 0, tau2 / (tau2 + sig2 / np.maximum(N, 1e-9)), 0.0)
        return m, mu + B * (m - mu), np.sqrt(B * sig2 / np.maximum(N, 1e-9))
    m_p, th_p, sd_p = shrink(Np, Vp, prm["mu_p"], prm["sig2_p"], prm["tau2_p"])
    m_n, th_n, _ = shrink(Nn, Vn, prm["mu_np"], prm["sig2_np"], prm["tau2_np"])
    both = (Np > 0) & (Nn > 0)
    var_d = prm["sig2_p"] / np.maximum(Np, 1e-9) + prm["sig2_np"] / np.maximum(Nn, 1e-9)
    B_d = np.where(both, prm["tau2_d"] / (prm["tau2_d"] + var_d), 0.0)
    lift_raw = np.where(both, m_p - m_n, prm["mu_d"])
    lift = prm["mu_d"] + B_d * (lift_raw - prm["mu_d"])
    has = (bp >= cmin).astype(float)
    pos = has * (vp > 0)
    n_has, n_pos = bc(has), bc(pos)
    cons = (n_pos + cprior * prm["p0"]) / (n_has + cprior)
    zp, zl, zc = _z(th_p, elig), _z(lift, elig), _z(cons, elig)
    rating = weights[0] * zp + weights[1] * zl + weights[2] * zc
    return {"Np": Np, "Nn": Nn, "m_p": m_p, "th_p": th_p, "sd_p": sd_p, "m_n": m_n, "th_n": th_n,
            "lift": lift, "lift_raw": lift_raw, "cons": cons, "n_has": n_has, "n_pos": n_pos,
            "zp": zp, "zl": zl, "zc": zc, "rating": rating}


def rate_table(pm, prm, cfg, min_balls, min_matches, weights=None, fixed_eligible=None) -> pd.DataFrame:
    r = cfg["rating"]
    if weights is None:
        weights = [r["weights"]["pressure"], r["weights"]["lift"], r["weights"]["consistency"]]
    players, pidx = np.unique(pm["player_id"].to_numpy(), return_inverse=True)
    n = len(players)
    bp, vp, bnp, vnp = (pm[c].to_numpy(dtype=float) for c in ["bp", "vp", "bnp", "vnp"])
    w = np.ones(len(pm))
    matches_p = np.bincount(pidx, weights=(bp > 0).astype(float), minlength=n)
    Np = np.bincount(pidx, weights=bp, minlength=n)
    elig = (Np >= min_balls) & (matches_p >= min_matches)
    if fixed_eligible is not None:
        elig = np.isin(players, list(fixed_eligible))
    a = rate_arrays(pidx, n, bp, vp, bnp, vnp, w, prm, weights, elig,
                    r["consistency_min_balls"], r["consistency_prior_matches"])
    t = pd.DataFrame({
        "player_id": players, "eligible": elig,
        "pressure_balls": a["Np"], "nonpressure_balls": a["Nn"], "pressure_matches": matches_p,
        "pressure_value_per_ball_raw": np.bincount(pidx, weights=pm["raw_vp"].to_numpy(), minlength=n) / np.maximum(a["Np"], 1),
        "pressure_value_per_ball_adj": a["m_p"], "pressure_value_per_ball_shrunk": a["th_p"],
        "pressure_value_ci_low": a["th_p"] - 1.96 * a["sd_p"], "pressure_value_ci_high": a["th_p"] + 1.96 * a["sd_p"],
        "nonpressure_value_per_ball_adj": a["m_n"], "nonpressure_value_per_ball_shrunk": a["th_n"],
        "lift_unshrunk": a["lift_raw"], "lift": a["lift"],
        "consistency": a["cons"], "consistency_matches": a["n_has"], "positive_matches": a["n_pos"],
        "z_pressure": a["zp"], "z_lift": a["zl"], "z_consistency": a["zc"], "rating": a["rating"],
        "pressure_wpa_total": np.bincount(pidx, weights=pm["wpa_p"].to_numpy(), minlength=n),
    })
    t["pressure_wpa_per_100_balls"] = 100 * t["pressure_wpa_total"] / t["pressure_balls"].clip(lower=1)
    t["rank"] = t["rating"].rank(ascending=False, method="min")
    return t.sort_values(["rating"], ascending=False, na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------- descriptive stats
def _trad_stats(rec, role):
    p = rec["pressure"].to_numpy()
    if role == "bat":
        f = lambda d, s: pd.DataFrame({
            f"{s}_runs": d.groupby("player_id")["runs"].sum(),
            f"{s}_outs": d.groupby("player_id")["outs"].sum(),
            f"{s}_strike_rate": 100 * d.groupby("player_id")["runs"].sum() / d.groupby("player_id")["balls"].sum().clip(lower=1),
            f"{s}_boundary_pct": 100 * (d.groupby("player_id")["fours"].sum() + d.groupby("player_id")["sixes"].sum())
            / d.groupby("player_id")["balls"].sum().clip(lower=1)})
        t = f(rec[p], "pressure").join(f(rec[~p], "nonpressure")[["nonpressure_strike_rate"]], how="outer")
        t["pressure_average"] = t["pressure_runs"] / t["pressure_outs"].clip(lower=1)
    else:
        f = lambda d, s: pd.DataFrame({
            f"{s}_runs_conceded": d.groupby("player_id")["runs"].sum(),
            f"{s}_wickets": d.groupby("player_id")["wickets"].sum(),
            f"{s}_economy": 6 * d.groupby("player_id")["runs"].sum() / d.groupby("player_id")["balls"].sum().clip(lower=1),
            f"{s}_dot_pct": 100 * d.groupby("player_id")["dots"].sum() / d.groupby("player_id")["balls"].sum().clip(lower=1)})
        t = f(rec[p], "pressure").join(f(rec[~p], "nonpressure")[["nonpressure_economy"]], how="outer")
        t["pressure_balls_per_wicket"] = (rec[p].groupby("player_id")["balls"].sum() / t["pressure_wickets"].clip(lower=1))
    return t


def _teams_for(rec, team_names):
    f = lambda ids: ", ".join(sorted({str(team_names.get(i, i)) for i in ids}))
    return rec.groupby("player_id")["team_id"].agg(lambda x: f(x.unique())).rename("teams")


def build_ratings(val: dict, data: dict, cfg: dict) -> dict:
    bat, bowl = val["bat"], val["bowl"]
    strengths = opposition_strength(bat, bowl, cfg)
    bat, bowl = apply_opposition(bat, bowl, strengths, cfg)
    e, out = cfg["eligibility"], {"strengths": strengths}
    for role, rec, min_balls in [("bat", bat, e["batter_min_pressure_balls"]), ("bowl", bowl, e["bowler_min_pressure_balls"])]:
        pm = player_match(rec)
        prm = fit_params(rec, pm, "pressure", cfg)
        tab = rate_table(pm, prm, cfg, min_balls, e["min_pressure_matches"])
        # competition-specific ratings with scaled thresholds
        for ct in cfg["competition"]["rated_types"]:
            sub_rec, sub_pm = rec[rec["comp_type"] == ct], pm[pm["comp_type"] == ct]
            if len(sub_pm) == 0:
                continue
            sp = fit_params(sub_rec, sub_pm, "pressure", cfg)
            st = rate_table(sub_pm, sp, cfg, min_balls * e["split_scale"], e["min_pressure_matches"] * e["split_scale"])
            st = st.set_index("player_id")
            pre = ct.lower()
            tab[f"{pre}_pressure_balls"] = tab["player_id"].map(st["pressure_balls"]).fillna(0)
            tab[f"{pre}_value_per_ball_shrunk"] = tab["player_id"].map(st["pressure_value_per_ball_shrunk"])
            tab[f"{pre}_rating"] = tab["player_id"].map(st["rating"])
            tab[f"{pre}_rank"] = tab["player_id"].map(st["rank"])
            out[f"{role}_{pre}_eligible"] = int(st["eligible"].sum())
        tab = tab.join(_trad_stats(rec, role), on="player_id")
        tab.insert(1, "player_name", tab["player_id"].map(data["names"]))
        tab.insert(2, "teams", tab["player_id"].map(_teams_for(rec, data["team_names"])))
        out[role] = {"rec": rec, "pm": pm, "prm": prm, "table": tab, "min_balls": min_balls,
                     "n_players": len(tab), "n_eligible": int(tab["eligible"].sum())}
    return out


def write_tables(ratings: dict, cfg: dict):
    od = cfg["paths"]["output_dir"]
    os.makedirs(od, exist_ok=True)
    for role, name in [("bat", "batters"), ("bowl", "bowlers")]:
        t = ratings[role]["table"]
        first = ["rank", "player_id", "player_name", "teams", "rating"]
        t = t[first + [c for c in t.columns if c not in first]]
        ratings[role]["table"] = t
        el = t[t["eligible"]]
        el.round(5).to_csv(os.path.join(od, f"{name}_ranked_full.csv"), index=False)
        t.round(5).to_csv(os.path.join(od, f"{name}_all_players.csv"), index=False)
        el.head(10).round(5).to_csv(os.path.join(od, f"top10_{name}.csv"), index=False)
