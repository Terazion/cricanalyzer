"""Phase 6: reliability checks - split-half, match bootstrap, sensitivity, season stability."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rating import fit_params, rate_arrays, rate_table
from .value import player_match


def _weights(cfg):
    w = cfg["rating"]["weights"]
    return [w["pressure"], w["lift"], w["consistency"]]


def split_half(role: dict, cfg: dict) -> dict:
    pm, prm = role["pm"], role["prm"]
    elig = set(role["table"].loc[role["table"]["eligible"], "player_id"])
    order = pm.drop_duplicates("match_id").sort_values(["match_dt", "match_id"])["match_id"].to_numpy()
    half = {m: i % 2 for i, m in enumerate(order)}
    h = pm["match_id"].map(half).to_numpy()
    tabs = [rate_table(pm[h == k], prm, cfg, 0, 0, fixed_eligible=elig).set_index("player_id") for k in (0, 1)]
    cols = {"rating": "rating", "pressure_value": "pressure_value_per_ball_shrunk",
            "pressure_value_unshrunk": "pressure_value_per_ball_adj", "lift": "lift", "consistency": "consistency"}
    j = tabs[0][list(cols.values())].join(tabs[1][list(cols.values())], lsuffix="_odd", rsuffix="_even", how="inner").dropna()
    rows = []
    for name, c in cols.items():
        r = j[f"{c}_odd"].corr(j[f"{c}_even"]) if len(j) > 2 else np.nan
        rho = j[f"{c}_odd"].corr(j[f"{c}_even"], method="spearman") if len(j) > 2 else np.nan
        rows.append({"component": name, "pearson_r": r, "spearman_rho": rho,
                     "spearman_brown_full_length": 2 * r / (1 + r) if pd.notna(r) and r > -1 else np.nan,
                     "players": len(j)})
    return {"table": pd.DataFrame(rows), "pairs": j[["rating_odd", "rating_even"]].reset_index()}


def bootstrap(role: dict, cfg: dict) -> pd.DataFrame:
    """Resample matches with replacement; rank the fixed eligible pool in each resample."""
    pm, prm, tab = role["pm"], role["prm"], role["table"]
    r = cfg["rating"]
    B = cfg["reliability"]["bootstrap_resamples"]
    rng = np.random.default_rng(cfg["seed"] + 7)
    players, pidx = np.unique(pm["player_id"].to_numpy(), return_inverse=True)
    matches, midx = np.unique(pm["match_id"].to_numpy(), return_inverse=True)
    elig = np.isin(players, tab.loc[tab["eligible"], "player_id"])
    bp, vp, bnp, vnp = (pm[c].to_numpy(dtype=float) for c in ["bp", "vp", "bnp", "vnp"])
    n_el = int(elig.sum())
    ratings = np.full((B, n_el), np.nan)
    ranks = np.full((B, n_el), np.nan)
    for b in range(B):
        counts = np.bincount(rng.integers(0, len(matches), len(matches)), minlength=len(matches))
        a = rate_arrays(pidx, len(players), bp, vp, bnp, vnp, counts[midx].astype(float), prm, _weights(cfg),
                        elig, r["consistency_min_balls"], r["consistency_prior_matches"])
        rt = a["rating"][elig]
        ratings[b] = rt
        ranks[b] = (-rt).argsort().argsort() + 1
    q = lambda x, p: np.nanpercentile(x, p, axis=0)
    return pd.DataFrame({
        "player_id": players[elig],
        "rating_boot_low": q(ratings, 2.5), "rating_boot_high": q(ratings, 97.5),
        "rank_boot_low": q(ranks, 2.5), "rank_boot_median": q(ranks, 50), "rank_boot_high": q(ranks, 97.5),
        "prob_top10": (ranks <= 10).mean(axis=0), "prob_top20": (ranks <= 20).mean(axis=0)})


def sensitivity(role: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    rec, base = role["rec"], role["table"]
    e = cfg["eligibility"]
    base_top = list(base.loc[base["eligible"], "player_id"].head(10))
    base_rating = base.set_index("player_id")["rating"]
    base_share = rec.loc[rec["balls"] > 0, "pressure"].mean()
    rows, tops = [], {}

    def add(label, tab):
        el = tab[tab["eligible"]]
        top = list(el["player_id"].head(10))
        tops[label] = top
        both = el.set_index("player_id")["rating"].to_frame("v").join(base_rating.rename("b"), how="inner").dropna()
        rows.append({"variant": label, "eligible_players": len(el),
                     "top10_overlap_with_base": len(set(top) & set(base_top)),
                     "base_top10_median_rank": float(el.set_index("player_id")["rank"].reindex(base_top).median()),
                     "spearman_vs_base": both["v"].corr(both["b"], method="spearman") if len(both) > 2 else np.nan})

    flags = [(f"top {int(round(s * 100))}% leverage", f"pressure_{int(round(s * 100))}")
             for s in cfg["reliability"]["sensitivity_thresholds"]] + [("rule-based pressure", "pressure_rule")]
    for label, flag in flags:
        if flag not in rec.columns:
            continue
        share = rec.loc[rec["balls"] > 0, flag].mean()
        scale = share / base_share if cfg["reliability"]["scale_eligibility_with_threshold"] and base_share > 0 else 1.0
        pm = player_match(rec, flag)
        prm = fit_params(rec, pm, flag, cfg)
        add(label, rate_table(pm, prm, cfg, role["min_balls"] * scale, e["min_pressure_matches"] * min(scale, 1.0)))
    for label, w in cfg["reliability"]["alternative_weights"].items():
        add(f"weights {label} {tuple(w)}", rate_table(role["pm"], role["prm"], cfg, role["min_balls"],
                                                      e["min_pressure_matches"], weights=w))
    raw = rec.assign(adj_value=rec["value"])
    pm = player_match(raw)
    add("no opposition adjustment", rate_table(pm, fit_params(raw, pm, "pressure", cfg), cfg,
                                               role["min_balls"], e["min_pressure_matches"]))
    return pd.DataFrame(rows), tops


def season_stability(role: dict, cfg: dict, top_n: int = 10) -> pd.DataFrame:
    """Per-season pressure value (shrunk with the global prior) for the top players."""
    tab, pm, prm = role["table"], role["pm"], role["prm"]
    top = tab[tab["eligible"]].head(top_n)
    g = pm[pm["player_id"].isin(top["player_id"])].groupby(["player_id", "season"])[["bp", "vp"]].sum().reset_index()
    n = g["bp"].clip(lower=1e-9)
    B = prm["tau2_p"] / (prm["tau2_p"] + prm["sig2_p"] / n)
    g["value_per_ball_adj"] = g["vp"] / n
    g["value_per_ball_shrunk"] = prm["mu_p"] + B * (g["value_per_ball_adj"] - prm["mu_p"])
    g = g.rename(columns={"bp": "pressure_balls"}).drop(columns="vp")
    g["player_name"] = g["player_id"].map(top.set_index("player_id")["player_name"])
    g["above_population_mean"] = g["value_per_ball_shrunk"] > prm["mu_p"]
    return g[g["pressure_balls"] > 0]


def run_all(ratings: dict, cfg: dict) -> dict:
    out = {}
    top_n = cfg["reliability"]["bootstrap_top_n"]
    for role in ["bat", "bowl"]:
        R = ratings[role]
        sh = split_half(R, cfg)
        bs = bootstrap(R, cfg)
        R["table"] = R["table"].merge(bs, on="player_id", how="left")
        sens, tops = sensitivity(R, cfg)
        el = R["table"][R["table"]["eligible"]]
        out[role] = {"split_half": sh, "sensitivity": sens, "sensitivity_tops": tops,
                     "season": season_stability(R, cfg),
                     "boot_top": el.head(top_n)[["rank", "player_id", "player_name", "rating", "rating_boot_low",
                                                 "rating_boot_high", "rank_boot_low", "rank_boot_median",
                                                 "rank_boot_high", "prob_top10"]],
                     "wpa_spearman": el["pressure_value_per_ball_shrunk"].corr(el["pressure_wpa_per_100_balls"], method="spearman")
                     if len(el) > 2 else np.nan}
    return out
