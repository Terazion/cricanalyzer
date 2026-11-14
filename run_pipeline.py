"""One command reproduces everything:  python run_pipeline.py

Optional: --config PATH --data-dir DIR --output-dir DIR --bootstrap N
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd

from cricanalyzer import load, models, pressure, rating, report, state, validate, value


def banner(t):
    print(f"\n=== {t} " + "=" * max(4, 70 - len(t)), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir")
    ap.add_argument("--output-dir")
    ap.add_argument("--bootstrap", type=int)
    args = ap.parse_args(argv)
    cfg = load.load_config(args.config)
    if args.data_dir:
        cfg["paths"]["data_dir"] = args.data_dir
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir
    if args.bootstrap:
        cfg["reliability"]["bootstrap_resamples"] = args.bootstrap
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    od = cfg["paths"]["output_dir"]
    td = os.path.join(od, "tables")
    os.makedirs(td, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    t0 = time.time()

    # ------------------------------------------------------------ Phase 1
    banner("PHASE 1 - MATCH STATE")
    data = load.load_data(cfg)
    m = data["meta"]
    print(f"ball_by_ball rows: {m['raw_rows']:,} ({m['raw_matches']} matches) -> {m['rows_regulation']:,} regulation rows")
    print(f"matches.csv rows: {m['matches_file_rows']}; without ball data (ignored): {m['matches_without_ball_data']}")
    df = state.build_states(data["bbb"], data["matches"], cfg)
    try:
        chk = state.run_state_checks(df, data["matches"], cfg)
    except state.StateCheckError as e:
        print(str(e))
        print("\nStopping: fix the data or rules before any rating is produced.")
        return 1
    mt = data["matches"]
    print(f"checks passed: {chk['run_margin_matches_checked']} 'won by N runs' margins reproduced exactly; "
          f"max legal balls {chk['max_legal_balls']}; max wickets {chk['max_wickets']}")
    print("matches by competition type:", mt["comp_type"].value_counts().to_dict())
    print(f"DLS matches: {int(mt['is_dls'].sum())} (without revised target: {int((mt['is_dls'] & mt['revised_target'].isna()).sum())}); "
          f"no winner: {int(mt['win_team_id'].isna().sum())}; super-over ties: {int(mt['has_super_over'].sum())}")
    print(f"legal balls: {int(df['is_legal'].sum()):,}; wickets: {int(df['is_wicket'].sum()):,}; "
          f"bowler-credited: {int(df['bowler_wicket'].sum()):,}; names shared by >1 player_id: {m['name_collisions']}")

    # ------------------------------------------------------------ Phase 2
    banner("PHASE 2 - BASELINE MODELS")
    mod = models.fit_all(df, cfg)
    print(f"split by match: {mod['n_train_matches']} train / {mod['n_test_matches']} test")
    print(mod["comparison"].round(5).to_string(index=False))
    print("\nwin probability (held-out matches):")
    print(mod["wp_metrics"].round(4).to_string(index=False))
    print(f"kept WP model: {mod['wp_method']}")
    mod["comparison"].to_csv(os.path.join(td, "model_comparison.csv"), index=False)
    mod["wp_metrics"].to_csv(os.path.join(td, "wp_metrics.csv"), index=False)
    mod["calibration"].to_csv(os.path.join(td, "wp_calibration.csv"), index=False)

    # ------------------------------------------------------------ Phase 3
    banner("PHASE 3 - PRESSURE INDEX")
    dfl = pressure.compute_leverage(mod["df"], mod["wp_models"], cfg)
    rated_types = cfg["competition"]["rated_types"] if cfg["competition"]["exclude_other_from_ratings"] else ["IPL", "T20I", "OTHER"]
    rated = dfl["comp_type"].isin(rated_types).to_numpy()
    dfl, thresholds = pressure.flag_pressure(dfl, cfg, rated)
    ev = pressure.evidence(dfl)
    sc = dfl[dfl["scored"]]
    print(f"scored deliveries (rated competitions, valid state): {len(sc):,} in {sc['match_id'].nunique()} matches")
    print("leverage thresholds by share:", {k: round(v, 5) for k, v in thresholds.items()})
    print(f"pressure balls: {int(sc['pressure'].sum()):,} ({100 * sc['pressure'].mean():.1f}%); "
          f"rule-based: {100 * ev['rule_share']:.1f}%; agreement {100 * ev['rule_agreement']:.1f}%")
    print("pressure share by innings:", {int(k): round(v, 3) for k, v in ev["share_inn"].items()})
    print(ev["league"].round(2).to_string(index=False))
    for k in ["by_over", "league", "by_phase", "examples_high", "examples_low"]:
        ev[k].to_csv(os.path.join(td, f"pressure_{k}.csv"), index=False)

    # ------------------------------------------------------------ Phase 4
    banner("PHASE 4 - PLAYER VALUE")
    so = None
    if len(data["super_overs"]):
        so = state.build_super_over_states(data["super_overs"], data["matches"], cfg)
        so = models.predict_baselines(mod, so[so["comp_type"].isin(rated_types)])
    val = value.compute_values(dfl, cfg, so)
    n_so_all = 0 if so is None else len(so)
    print(f"super-over deliveries in rated competitions: {n_so_all} in {0 if so is None else so['match_id'].nunique()} matches; "
          f"added as pressure balls: {val['super_over_deliveries']}")
    print(f"WP per run: {val['wp_per_run']:.5f}; mean wicket cost {val['mean_wicket_cost']:.2f} runs "
          f"(pressure balls {val['mean_wicket_cost_pressure']:.2f}; capped on {100 * val['share_cost_capped']:.1f}% of balls)")
    print(f"pressure prior: {cfg['rating'].get('pressure_prior', 'population')}")
    print(f"wicket cost conversion: {val['conversion']}; cost quantiles (runs): {val['cost_quantiles']}")
    print(val["diagnostics"].set_index("segment").T.round(4).to_string())
    val["diagnostics"].to_csv(os.path.join(td, "value_diagnostics.csv"), index=False)
    print(f"batter records: {len(val['bat']):,} ({val['bat']['player_id'].nunique()} batters), mean value/ball "
          f"{val['bat']['value'].sum() / val['bat']['balls'].sum():+.4f}")
    print(f"bowler records: {len(val['bowl']):,} ({val['bowl']['player_id'].nunique()} bowlers), mean value/ball "
          f"{val['bowl']['value'].sum() / val['bowl']['balls'].sum():+.4f}")

    # ------------------------------------------------------------ Phase 5
    banner("PHASE 5 - RATING")
    rt = rating.build_ratings(val, data, cfg)
    for role, name in [("bat", "batters"), ("bowl", "bowlers")]:
        R = rt[role]
        print(f"{name}: {R['n_eligible']} eligible of {R['n_players']} "
              f"(player-match rows {len(R['pm']):,}); EB: sd/ball {np.sqrt(R['prm']['sig2_p']):.2f}, "
              f"between-player sd {np.sqrt(R['prm']['tau2_p']):.3f}")
        if R["prm"]["tau2_p_raw"] <= 0:
            print(f"  WARNING: no detectable between-player variance in pressure value for {name}; "
                  "ratings are dominated by noise")
        if R["n_eligible"] < 10:
            print(f"  WARNING: fewer than 10 eligible {name}; lower the eligibility thresholds in config.yaml")
    rt["strengths"].reset_index().to_csv(os.path.join(td, "team_season_strength.csv"), index=False)

    # ------------------------------------------------------------ Phase 6
    banner("PHASE 6 - RELIABILITY")
    rel = validate.run_all(rt, cfg)
    rating.write_tables(rt, cfg)
    for role, name in [("bat", "batters"), ("bowl", "bowlers")]:
        R = rel[role]
        sh = R["split_half"]["table"].set_index("component")
        print(f"{name}: split-half r = {sh.loc['rating', 'pearson_r']:.3f} (n={int(sh.loc['rating', 'players'])}); "
              f"value-vs-WPA Spearman {R['wpa_spearman']:.2f}")
        print(R["split_half"]["table"].round(3).to_string(index=False))
        print(R["sensitivity"][["variant", "eligible_players", "top10_overlap_with_base", "spearman_vs_base"]].round(2).to_string(index=False))
        R["split_half"]["table"].to_csv(os.path.join(td, f"split_half_{name}.csv"), index=False)
        R["sensitivity"].to_csv(os.path.join(td, f"sensitivity_{name}.csv"), index=False)
        R["season"].to_csv(os.path.join(td, f"season_stability_{name}.csv"), index=False)
        R["boot_top"].to_csv(os.path.join(td, f"bootstrap_top20_{name}.csv"), index=False)

    soe = validate.super_over_effect(dfl, data, cfg, rt, so) if so is not None and len(so) else None
    if soe is not None:
        soe.to_csv(os.path.join(td, "super_over_effect.csv"), index=False)
        print("\nranking vs the other super-over setting (include_in_ratings flipped):")
        print(soe.round(3).to_string(index=False))
    mc = validate.method_comparison(dfl, data, cfg, rt, so)
    mc.to_csv(os.path.join(td, "method_comparison.csv"), index=False)
    print("\nmethod comparison (split-half r; higher = more repeatable):")
    print(mc.round(3).to_string(index=False))

    # ------------------------------------------------------------ Phase 7
    banner("PHASE 7 - REPORT")
    ctx = {"meta": m, "models": mod, "evidence": ev, "thresholds": thresholds, "val": val, "ratings": rt,
           "rel": rel, "method_comparison": mc, "super_over_effect": soe, "n_super_over": n_so_all, "team_names": data["team_names"]}
    path = report.write_report(ctx, cfg)
    print(f"report for upload: {path[:-3]}.html (open in a browser, Print > Save as PDF)")
    print(f"report: {path}; charts: {len(os.listdir(os.path.join(od, 'charts')))} PNG files")
    for role, name in [("bat", "BATTERS"), ("bowl", "BOWLERS")]:
        t = rt[role]["table"]
        cols = ["rank", "player_id", "player_name", "rating", "pressure_balls", "rank_boot_low", "rank_boot_high"]
        print(f"\nTOP 10 {name} UNDER PRESSURE")
        print(t[t["eligible"]].head(10)[cols].round(2).to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in '{od}/'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
