"""Phase 7: charts (PNG) and the methodology report (markdown)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
    "axes.axisbelow": True,
})


def _save(fig, cdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(cdir, name))
    plt.close(fig)
    return f"charts/{name}"


# ----------------------------------------------------------------- markdown helpers
def _fmt(v, nd=3):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "-"
        if float(v).is_integer() and abs(v) < 1e9:
            return str(int(v))
        return f"{v:.{nd}f}"
    return str(v)


def md_table(df: pd.DataFrame, nd=3, rename=None) -> str:
    df = df.rename(columns=rename or {})
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(_fmt(v, nd) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


# ----------------------------------------------------------------- charts
def make_charts(ctx: dict, cfg: dict) -> dict:
    cdir = os.path.join(cfg["paths"]["output_dir"], "charts")
    os.makedirs(cdir, exist_ok=True)
    ch = {}
    mod, ev, df = ctx["models"], ctx["evidence"], ctx["val"]["df"]

    # calibration
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
    for lab, col, name in [("both", BLUE, "both innings"), ("1", ORANGE, "innings 1"), ("2", AQUA, "innings 2")]:
        c = mod["calibration"][mod["calibration"]["innings"] == lab]
        ax.plot(c["predicted"], c["observed"], marker="o", ms=5, lw=2, color=col, label=name)
    ax.set(xlabel="Predicted win probability (batting side)", ylabel="Observed win share",
           title="Win probability calibration, held-out matches", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    ch["calibration"] = _save(fig, cdir, "wp_calibration.png")

    # pressure share by over
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bo = ev["by_over"]
    for col, c, name in [("innings_1", BLUE, "Innings 1"), ("innings_2", ORANGE, "Innings 2")]:
        if col in bo:
            ax.plot(bo["overid"], 100 * bo[col], color=c, lw=2, marker="o", ms=4, label=name)
    ax.axvspan(0.5, 6.5, color=GRID, alpha=0.45, lw=0)
    ax.axvspan(15.5, 20.5, color=GRID, alpha=0.45, lw=0)
    ax.set(xlabel="Over (shaded: powerplay and death)", ylabel="% of deliveries flagged as pressure",
           title="Where pressure balls occur", xticks=range(1, 21), xlim=(0.5, 20.5))
    ax.set_ylim(bottom=0)
    ax.legend()
    ch["by_over"] = _save(fig, cdir, "pressure_share_by_over.png")

    # leverage distribution
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    lev = df["leverage"].dropna()
    ax.hist(lev[lev <= lev.quantile(0.995)], bins=80, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
    thr = ctx["thresholds"][cfg["pressure"]["top_share"]]
    ax.axvline(thr, color=INK, lw=1.2, ls="--")
    ax.text(thr, ax.get_ylim()[1] * 0.92, f"  pressure threshold ({thr:.4f})", color=INK, va="top")
    ax.set(xlabel="Leverage: expected |change in win probability| from one ball", ylabel="Deliveries",
           title="Leverage distribution (IPL + T20I)")
    ch["leverage"] = _save(fig, cdir, "leverage_distribution.png")

    # league-wide pressure vs non-pressure
    lg = ev["league"].set_index("segment")
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    for ax, (col, title) in zip(axes, [("strike_rate", "Strike rate"), ("dismissal_rate_per_100_balls", "Dismissals per 100 balls"),
                                       ("economy", "Economy (runs/over)"), ("dot_pct", "Dot-ball %")]):
        vals = [lg.loc[s, col] if s in lg.index else np.nan for s in ["non-pressure", "pressure"]]
        bars = ax.bar(["Non-pressure", "Pressure"], vals, color=[BLUE, ORANGE], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, np.nanmax(vals) * 1.18)
    ch["league"] = _save(fig, cdir, "pressure_vs_nonpressure.png")

    for role, name in [("bat", "batters"), ("bowl", "bowlers")]:
        tab = ctx["ratings"][role]["table"]
        el = tab[tab["eligible"]]
        top = el.head(10).iloc[::-1]
        lab = lambda t: [f"{int(r)}. {n}" for r, n in zip(t["rank"], t["player_name"])]

        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.barh(lab(top), top["rating"], color=BLUE, height=0.6)
        if "rating_boot_low" in top:
            ax.errorbar(top["rating"], lab(top), xerr=[(top["rating"] - top["rating_boot_low"]).clip(lower=0), (top["rating_boot_high"] - top["rating"]).clip(lower=0)],
                        fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
        ax.grid(axis="y", visible=False)
        ax.set(xlabel="Pressure rating (weighted z-score; bars = 95% match-bootstrap interval)",
               title=f"Top 10 {name} under pressure")
        ch[f"top10_{role}"] = _save(fig, cdir, f"top10_{name}.png")

        fig, ax = plt.subplots(figsize=(8, 4.6))
        comp = top[["z_pressure", "z_lift", "z_consistency"]].to_numpy()
        y = np.arange(len(top))
        for j, (c, l) in enumerate([(BLUE, "Pressure value (z)"), (ORANGE, "Lift vs own baseline (z)"), (AQUA, "Consistency (z)")]):
            ax.barh(y + (1 - j) * 0.26, comp[:, j], height=0.24, color=c, label=l)
        ax.set_yticks(y, lab(top))
        ax.axvline(0, color=MUTED, lw=1)
        ax.grid(axis="y", visible=False)
        ax.set(xlabel="z-score within eligible pool", title=f"Rating components, top 10 {name}")
        ax.title.set_position((0, 1.12)); ax.set_title(f"Rating components, top 10 {name}", pad=30)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=3, fontsize=8)
        ch[f"components_{role}"] = _save(fig, cdir, f"rating_components_{name}.png")

        if "rank_boot_low" in el:
            t20 = el.head(cfg["reliability"]["bootstrap_top_n"]).iloc[::-1]
            fig, ax = plt.subplots(figsize=(8, 6.2))
            ax.hlines(lab(t20), t20["rank_boot_low"], t20["rank_boot_high"], color=BLUE, lw=3, alpha=0.45)
            ax.plot(t20["rank"], lab(t20), "o", color=BLUE, ms=6)
            ax.axvline(10.5, color=MUTED, lw=1, ls="--")
            ax.grid(axis="y", visible=False)
            ax.set(xlabel=f"Rank among {len(el)} eligible players (dot = actual, line = 95% bootstrap interval)",
                   title=f"Rank uncertainty, top {len(t20)} {name}")
            ax.set_xlim(left=0)
            ch[f"boot_{role}"] = _save(fig, cdir, f"bootstrap_rank_{name}.png")

        pairs = ctx["rel"][role]["split_half"]["pairs"]
        r = ctx["rel"][role]["split_half"]["table"].set_index("component").loc["rating", "pearson_r"]
        fig, ax = plt.subplots(figsize=(5.2, 5))
        ax.scatter(pairs["rating_odd"], pairs["rating_even"], s=22, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
        ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
        ax.set(xlabel="Rating from odd-numbered matches", ylabel="Rating from even-numbered matches",
               title=f"Split-half reliability, {name} (r = {r:.2f})")
        ch[f"split_{role}"] = _save(fig, cdir, f"split_half_{name}.png")

        ss = ctx["rel"][role]["season"]
        ids = list(el.head(10)["player_id"])
        if len(ss) and len(ids):
            mu = ctx["ratings"][role]["prm"]["mu_p"]
            fig, axes = plt.subplots(2, 5, figsize=(13, 5), sharex=True, sharey=True)
            for ax, pid in zip(axes.ravel(), ids):
                s = ss[ss["player_id"] == pid].sort_values("season")
                ax.axhline(mu, color=MUTED, lw=1, ls="--")
                ax.plot(s["season"], s["value_per_ball_shrunk"], color=BLUE, lw=2, marker="o", ms=5)
                ax.set_title(str(s["player_name"].iloc[0]) if len(s) else str(pid), fontsize=9)
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=4))
            for ax in axes.ravel()[len(ids):]:
                ax.axis("off")
            fig.suptitle(f"Season stability, top 10 {name}: shrunk pressure value per ball (dashed = population mean)",
                         x=0.01, ha="left", fontweight="bold")
            ch[f"season_{role}"] = _save(fig, cdir, f"season_stability_{name}.png")

    sb, sw = ctx["rel"]["bat"]["sensitivity"], ctx["rel"]["bowl"]["sensitivity"]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(sb) + 1.6))
    y = np.arange(len(sb))[::-1]
    ax.barh(y + 0.19, sb["top10_overlap_with_base"], height=0.36, color=BLUE, label="Batters")
    ax.barh(y - 0.19, sw["top10_overlap_with_base"], height=0.36, color=ORANGE, label="Bowlers")
    ax.set_yticks(y, sb["variant"])
    ax.set(xlim=(0, 10.8), xlabel="Players shared with the base top 10", title="Sensitivity: top-10 overlap by variant")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ch["sensitivity"] = _save(fig, cdir, "sensitivity_overlap.png")
    return ch


# ----------------------------------------------------------------- report
def write_report(ctx: dict, cfg: dict) -> str:
    ch = make_charts(ctx, cfg)
    od = cfg["paths"]["output_dir"]
    meta, mod, ev, val, rt, rel = ctx["meta"], ctx["models"], ctx["evidence"], ctx["val"], ctx["ratings"], ctx["rel"]
    df = val["df"]
    P, E, W = cfg["pressure"], cfg["eligibility"], cfg["rating"]["weights"]
    share = int(round(P["top_share"] * 100))
    L = []
    a = L.append
    img = lambda key, alt: a(f"\n![{alt}]({ch[key]})\n") if key in ch else None

    a("# Reliability under pressure in T20 cricket - methodology report")
    a("\n*SupeRR Selector Season 2, Round 1. Generated by `python run_pipeline.py`; every number below is "
      "computed from the supplied data with the settings in `config.yaml` (seed %d).*\n" % cfg["seed"])
    a("## Summary\n")
    a(f"- Data: {meta['rows_regulation']:,} regulation deliveries in {meta['matches']} matches "
      f"({meta['matches_without_ball_data']} matches without ball data ignored; super overs excluded). "
      f"Player ratings use {len(df):,} deliveries from {df['match_id'].nunique()} "
      f"{' and '.join(cfg['competition']['rated_types'])} matches.")
    a(f"- A **pressure ball** is a delivery in the top {share}% of *leverage*: the expected absolute swing in win "
      f"probability from that one ball. Knock-out matches get a x{P['knockout_multiplier']} multiplier.")
    a("- Players are valued in **runs relative to what an average player does in the same match state**, "
      "adjusted for opposition strength and shrunk toward the population mean for small samples.")
    a(f"- Rating = {W['pressure']} x pressure value + {W['lift']} x lift over own non-pressure baseline + "
      f"{W['consistency']} x consistency, each as a z-score among eligible players.")
    a(f"- Eligible: {rt['bat']['n_eligible']} batters (of {rt['bat']['n_players']}) and "
      f"{rt['bowl']['n_eligible']} bowlers (of {rt['bowl']['n_players']}).\n")

    for role, name, cols in [
        ("bat", "batters", ["rank", "player_id", "player_name", "teams", "rating", "pressure_balls", "pressure_matches",
                            "pressure_runs", "pressure_strike_rate", "pressure_outs"]),
        ("bowl", "bowlers", ["rank", "player_id", "player_name", "teams", "rating", "pressure_balls", "pressure_matches",
                             "pressure_economy", "pressure_wickets", "pressure_dot_pct"])]:
        t = rt[role]["table"]
        a(f"### Top 10 {name} under pressure\n")
        a(md_table(t[t["eligible"]].head(10)[cols], nd=2))
        a("")

    # ---------------- 1
    a("## 1. Pressure definition\n")
    a("**Idea.** Pressure is not a phase of the game or a scoreboard rule; it is how much the result depends on "
      "the next ball. We measure that directly with a win-probability model.\n")
    a("**Leverage of a delivery** = sum over outcomes o in {dot, 1, 2, 4, 6, wicket} of "
      "P(o | state) x |WP(state after o) - WP(state)|. "
      "P(o | state) comes from the baseline outcome model, WP from the win-probability model, both evaluated on the "
      "state *before* the ball (innings, score, wickets, balls remaining, target). At the end of the first innings "
      "the win probability is handed over to the chase model, so first-innings balls are scored on the same scale.\n")
    a(f"**Pressure ball** = leverage in the top {share}% of all rated deliveries "
      f"(threshold {ctx['thresholds'][P['top_share']]:.4f} in win-probability units). Leverage is multiplied by "
      f"{P['knockout_multiplier']} in knock-out matches before thresholding.\n")
    a("**Win probability model.** Trained on matches with a clear regulation result (ties = 0.5, no-results and "
      "DLS matches without a revised target excluded), split by match. Held-out performance:\n")
    a(md_table(mod["wp_metrics"], nd=4))
    a(f"\nModel kept: **{mod['wp_method']}** (lower held-out log loss). The gradient-boosted version is constrained "
      "to be monotone in runs, wickets and balls so that hypothetical outcomes move win probability in the "
      "right direction.\n")
    img("calibration", "Win probability calibration")
    a("**Is the definition meaningful?**\n")
    si = ev["share_inn"]
    a("- Share of deliveries flagged, by innings: " + ", ".join(f"innings {int(k)}: {100 * v:.1f}%" for k, v in si.items())
      + "; by competition: " + ", ".join(f"{k}: {100 * v:.1f}%" for k, v in ev["share_comp"].items())
      + "; knock-out vs other: " + ", ".join(f"{'knock-out' if k == 1 else 'other'}: {100 * v:.1f}%" for k, v in ev["share_ko"].items()) + ".")
    img("by_over", "Pressure share by over and innings")
    img("leverage", "Leverage distribution")
    a("League-wide behaviour on pressure vs non-pressure balls:\n")
    a(md_table(ev["league"], nd=2))
    img("league", "League-wide statistics in pressure vs non-pressure balls")
    a("By phase:\n")
    a(md_table(ev["by_phase"][["phase", "segment", "deliveries", "strike_rate", "dismissal_rate_per_100_balls", "economy"]], nd=2))
    a("\nHighest-leverage situations (one per match):\n")
    a(md_table(ev["examples_high"], nd=3))
    a("\nLowest-leverage situations (one per match):\n")
    a(md_table(ev["examples_low"], nd=5))
    rb = P["rule_based"]
    a(f"\n**Rule-based cross-check.** A simple scoreboard rule (chase with required rate >= {rb['chase_min_rrr']} and "
      f"<= {rb['chase_max_balls_left']} balls left; or {rb['powerplay_min_wickets']}+ wickets down in the powerplay; or death overs "
      f"of a chase with required rate between {rb['death_reach_min_rrr']} and {rb['death_reach_max_rrr']}) flags "
      f"{100 * ev['rule_share']:.1f}% of deliveries and agrees with the leverage flag on "
      f"{100 * ev['rule_agreement']:.1f}% of deliveries. Rankings under this rule are compared in section 5.\n")

    # ---------------- 2
    a("## 2. Performance measurement\n")
    a("**Baselines.** For every state we estimate what an average player produces: expected batter runs per ball "
      "faced, expected bowler-charged runs per delivery, dismissal probability, bowler-credited wicket probability, "
      "and the six-way outcome distribution. Inputs: innings, balls remaining, wickets down, required rate "
      "(second innings), competition type, season. Gradient boosting is compared with a hierarchically smoothed "
      "lookup table on held-out matches and the better one is kept:\n")
    a(md_table(mod["comparison"], nd=5))
    a("\nThe gain over a no-state baseline is small in absolute terms because single-ball outcomes are mostly noise; "
      "what matters is that the baseline removes the systematic part (phase, wickets in hand, chase situation). "
      f"All downstream predictions are cross-fitted ({cfg['models']['n_folds']} folds by match), so no delivery is "
      "judged by a model that saw its own match.\n")
    a("**Batter value per ball faced** = runs - expected runs(state) - (dismissed - P(dismissal | state)) x wicket cost. "
      "The dismissal is charged to `dismissed_player_id`, so a run-out of the non-striker is charged to the "
      "non-striker. Wides are not balls faced.\n")
    a("**Bowler value per delivery** = expected bowler-charged runs(state) - bowler-charged runs "
      "+ (bowler-credited wicket - P(bowler wicket | state)) x wicket cost. Byes, leg-byes and penalties are not charged; "
      "run-outs are not credited.\n")
    a(f"**Wicket cost in runs** = [WP(after a dot) - WP(after a wicket)] / (average WP per run), capped at "
      f"{cfg['value']['wicket_cost_cap_runs']} runs. Average WP per run = {val['wp_per_run']:.5f}; mean wicket cost "
      f"{val['mean_wicket_cost']:.2f} runs over all balls and {val['mean_wicket_cost_pressure']:.2f} runs on pressure balls "
      f"({100 * val['share_cost_capped']:.1f}% of balls hit the cap). Expected dismissals are credited back so an average "
      "player scores zero in every state.\n")
    a("**Cross-check: win probability added (WPA).** The actual change in win probability across each delivery is "
      "credited to the batter (or dismissed player) and debited to the bowler. Rank correlation between shrunk pressure value "
      f"and pressure WPA per 100 balls among eligible players: batters {rel['bat']['wpa_spearman']:.2f}, "
      f"bowlers {rel['bowl']['wpa_spearman']:.2f}.\n")

    # ---------------- 3
    a("## 3. Fair comparisons\n")
    a(f"- **Eligibility.** Batters need >= {E['batter_min_pressure_balls']} pressure balls faced, bowlers >= "
      f"{E['bowler_min_pressure_balls']} pressure balls bowled, and pressure balls in >= {E['min_pressure_matches']} matches. "
      f"{rt['bat']['n_eligible']} batters and {rt['bowl']['n_eligible']} bowlers qualify.")
    a("- **State adjustment.** Value is always relative to the expected outcome in the same state, so an opener is "
      "not compared with a finisher on raw strike rate, and a death bowler is not penalised for bowling the expensive overs.")
    for role, name in [("bat", "Batters"), ("bowl", "Bowlers")]:
        p = rt[role]["prm"]
        nb = p["sig2_p"] / p["tau2_p"]
        a(f"- **Shrinkage ({name.lower()}).** Per-ball spread sd = {np.sqrt(p['sig2_p']):.2f} runs; true between-player sd = "
          f"{np.sqrt(p['tau2_p']):.3f} runs per ball; a player needs about {nb:,.0f} pressure balls for the data to "
          "outweigh the prior. Estimates are pulled toward the population mean accordingly. "
          + ("**The variance estimate hit its floor: the data show no detectable between-player differences in pressure value, "
             "so this ranking is dominated by noise.** " if p.get("tau2_p_raw", 1.0) <= 0 else "") +
          f"Lift (pressure minus own non-pressure value) is shrunk as a difference; its estimated between-player sd is "
          f"{np.sqrt(max(p.get('tau2_d_raw', 0.0), 0.0)):.3f} runs per ball"
          + (" - **no detectable player-to-player variation in lift**, so this component is close to noise here."
             if p.get("tau2_d_raw", 0.0) <= 0 else "."))
    s = rt["strengths"]
    a(f"- **Opposition adjustment.** Batting and bowling strength of each team-season is estimated from the same value "
      f"metric, iterated so that strength is itself corrected for schedule, and shrunk toward zero "
      f"(prior {cfg['rating']['opposition']['prior_balls']} balls). Batter value is credited by the bowling strength faced and "
      f"vice versa. Bowling strength ranges from {s['bowl_strength'].min():+.3f} to {s['bowl_strength'].max():+.3f} "
      f"runs per ball across {len(s)} team-seasons; batting strength from {s['bat_strength'].min():+.3f} to {s['bat_strength'].max():+.3f}.")
    tn = ctx["team_names"]
    st = s.reset_index()
    st["team"] = st["team_id"].map(tn)
    a("\nStrongest and weakest bowling team-seasons (runs saved per ball vs average):\n")
    cs = ["team", "season", "bowl_strength", "bowl_balls", "bat_strength", "bat_balls"]
    a(md_table(pd.concat([st.sort_values("bowl_strength", ascending=False).head(5), st.sort_values("bowl_strength").head(5)])[cs], nd=3))
    a("\n- **IPL-only and T20I-only ratings** are reported next to the combined rating (thresholds scaled by "
      f"{E['split_scale']}), so a ranking that rests on one competition is visible:\n")
    for role, name in [("bat", "batters"), ("bowl", "bowlers")]:
        t = rt[role]["table"]
        cc = [c for c in ["rank", "player_name", "rating", "ipl_pressure_balls", "ipl_rating", "ipl_rank",
                          "t20i_pressure_balls", "t20i_rating", "t20i_rank"] if c in t.columns]
        a(f"Top 10 {name}:\n")
        a(md_table(t[t["eligible"]].head(10)[cc], nd=2))
        a("")

    # ---------------- 4
    a("## 4. Evidence for the top 10\n")
    for role, name, extra in [
        ("bat", "batters", ["pressure_runs", "pressure_outs", "pressure_average", "pressure_strike_rate", "nonpressure_strike_rate", "pressure_boundary_pct"]),
        ("bowl", "bowlers", ["pressure_runs_conceded", "pressure_wickets", "pressure_economy", "nonpressure_economy", "pressure_dot_pct", "pressure_balls_per_wicket"])]:
        t = rt[role]["table"]
        top = t[t["eligible"]].head(10)
        a(f"### {name.capitalize()}\n")
        img(f"top10_{role}", f"Top 10 {name}")
        a("Rating components (value in runs per ball, opposition-adjusted; CI = 95% empirical-Bayes interval):\n")
        a(md_table(top[["rank", "player_name", "rating", "pressure_balls", "pressure_value_per_ball_raw", "pressure_value_per_ball_adj",
                        "pressure_value_per_ball_shrunk", "pressure_value_ci_low", "pressure_value_ci_high",
                        "nonpressure_value_per_ball_shrunk", "lift_unshrunk", "lift", "consistency", "z_pressure", "z_lift", "z_consistency"]], nd=3,
                   rename={"pressure_value_per_ball_raw": "value/ball raw", "pressure_value_per_ball_adj": "value/ball adj",
                           "pressure_value_per_ball_shrunk": "value/ball shrunk", "pressure_value_ci_low": "CI low",
                           "pressure_value_ci_high": "CI high",
                           "nonpressure_value_per_ball_shrunk": "non-pressure value/ball", "lift_unshrunk": "lift raw"}))
        img(f"components_{role}", f"Rating components {name}")
        a("Conventional statistics on the same pressure balls:\n")
        a(md_table(top[["rank", "player_name", "teams", "pressure_matches"] + extra + ["pressure_wpa_total"]], nd=2))
        a("")

    # ---------------- 5
    a("## 5. Reliability and limitations\n")
    for role, name in [("bat", "Batters"), ("bowl", "Bowlers")]:
        R = rel[role]
        a(f"### {name}\n")
        a("**Split-half** (ratings recomputed separately on odd and even matches in date order):\n")
        a(md_table(R["split_half"]["table"], nd=3))
        img(f"split_{role}", f"Split-half {name}")
        a(f"**Bootstrap by match** ({cfg['reliability']['bootstrap_resamples']} resamples; models, opposition strengths and the eligible pool held fixed):\n")
        a(md_table(R["boot_top"], nd=2))
        img(f"boot_{role}", f"Bootstrap rank intervals {name}")
        a("**Sensitivity** (top-10 overlap with the base ranking):\n")
        a(md_table(R["sensitivity"], nd=2))
        a("\n**Season stability** (shrunk pressure value per ball by season, top 10):\n")
        ss = R["season"]
        if len(ss):
            piv = ss.pivot_table(index="player_name", columns="season", values="value_per_ball_shrunk")
            order = [n for n in rt[role]["table"][rt[role]["table"]["eligible"]].head(10)["player_name"] if n in piv.index]
            a(md_table(piv.loc[order].reset_index(), nd=3))
            above = ss.groupby("player_name")["above_population_mean"].agg(["sum", "count"])
            a("\nSeasons above the population mean: " + "; ".join(f"{n}: {int(above.loc[n, 'sum'])}/{int(above.loc[n, 'count'])}" for n in order) + ".")
        img(f"season_{role}", f"Season stability {name}")
        a("")
    img("sensitivity", "Sensitivity overlap")

    a("### Limitations\n")
    lim = [
        "**Small pressure samples.** Per-ball outcomes are extremely noisy relative to true skill differences (see the "
        "shrinkage numbers in section 3). Adjacent ranks are often not statistically distinguishable; read the bootstrap "
        "rank intervals, not just the rank.",
        "**Lift is the weakest component.** It is a difference of two noisy means; see its estimated between-player spread "
        "(section 3) and its split-half correlation below before giving it weight. The `pressure_only` sensitivity row shows the ranking without it.",
        "**Split-half correlations are the honest ceiling.** Where they are low, the rating is describing what happened "
        "in these matches more than a stable, repeatable trait.",
        "**Win probability ignores team quality, venue and toss.** It uses only the scoreboard, competition type and season, "
        "so leverage is slightly mis-stated when a strong side plays a weak one.",
        "**Leverage is unevenly spread across innings and overs** (section 1). Players whose balls fall mostly in low-leverage "
        "situations, or in one-sided games, have fewer pressure balls and may miss eligibility.",
        "**Opposition adjustment is coarse.** It is per team-season, not per bowler faced or batter bowled to, and associate "
        "sides that rarely play full members are only weakly connected to the rest of the schedule.",
        "**Credit assignment.** Batter value ignores the non-striker's contribution and running between wickets; bowler value "
        "ignores fielding (dropped catches, run-out assists) and treats all byes/leg-byes as nobody's fault.",
        "**Wicket cost uses one global runs-per-win-probability exchange rate** and a cap; in dead game states the cost goes to "
        "zero, which is correct for win probability but understates what a wicket is worth in runs.",
        "**DLS matches without revised targets** are dropped from second-innings scoring; rain-shortened first innings without "
        "revised-over fields are treated as full length.",
        "**Selection effects.** Captains choose who bowls and who bats in pressure moments; the method measures performance "
        "given the opportunity, not the opportunity itself. `bowl_speed`, venue effects and match-ups (pace/spin, handedness) are not used.",
        "**Bootstrap intervals hold the fitted models and opposition strengths fixed**, so they understate total uncertainty.",
        "**Rule-based and leverage definitions disagree on many balls**; the overlap table shows how much the top 10 depends on that choice.",
    ]
    for x in lim:
        a(f"- {x}")
    a("\n### Reproducibility\n")
    a(f"`python run_pipeline.py` regenerates every table and chart. Seed {cfg['seed']}; all thresholds and weights are in `config.yaml`. "
      "Outputs: `top10_batters.csv`, `top10_bowlers.csv`, `batters_ranked_full.csv`, `bowlers_ranked_full.csv` (eligible players, all components, "
      "sample sizes, intervals), `*_all_players.csv` (including non-eligible), and supporting tables in `outputs/tables/`.")
    path = os.path.join(od, "methodology_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path
