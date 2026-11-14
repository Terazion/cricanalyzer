# cricanalyzer

Ranks the most reliable T20 batters and bowlers under pressure (SupeRR Selector Season 2, Round 1).
Rankings come only from the ball-by-ball data; no reputational inputs.

## Run

    pip install -r requirements.txt
    # put the six CSV files in data/
    python run_pipeline.py          # everything: models, ratings, reliability, report, charts
    python -m pytest tests -q       # rule tests, real-data integrity tests, synthetic smoke test

CPU only, about a minute. Seeds are fixed (`seed` in `config.yaml`); two runs give identical outputs.
Every threshold and weight is in `config.yaml`.

## Outputs (`outputs/`)

| File | Content |
|---|---|
| `top10_batters.csv`, `top10_bowlers.csv` | ranked 1-10 |
| `batters_ranked_full.csv`, `bowlers_ranked_full.csv` | all eligible players: every component, sample sizes, EB and bootstrap intervals, IPL-only / T20I-only ratings |
| `*_all_players.csv` | same columns including non-eligible players |
| `methodology_report.html` | the report as one self-contained file for upload (open in a browser, Print > Save as PDF) |
| `methodology_report.md` | pressure definition, performance measurement, fair comparisons, evidence, reliability and limitations |
| `charts/*.png` | charts used by the report |
| `tables/*.csv` | model comparison, calibration, pressure evidence, sensitivity, split-half, bootstrap, season stability, team strengths |

## Layout

| Module | Phase |
|---|---|
| `cricanalyzer/load.py` | loading, verified delivery rules, canonical names |
| `cricanalyzer/state.py` | 1 - state before each ball, integrity checks (pipeline stops if any fails) |
| `cricanalyzer/models.py` | 2 - expected runs / wicket / outcome baselines (GB vs smoothed lookup), win probability |
| `cricanalyzer/pressure.py` | 3 - leverage, pressure flags, rule-based cross-check, evidence |
| `cricanalyzer/value.py` | 4 - batter and bowler value in runs, WPA |
| `cricanalyzer/rating.py` | 5 - opposition adjustment, empirical Bayes, composite rating, tables |
| `cricanalyzer/validate.py` | 6 - split-half, bootstrap, sensitivity, season stability |
| `cricanalyzer/report.py` | 7 - charts and markdown report |

`tests/synth.py` generates schema-compatible synthetic data for the smoke test only.
