"""Phase 1 integrity tests on the real data. Skipped when data/ is not present."""
import os

import pytest

from cricanalyzer import load, state

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "ball_by_ball.csv")),
                                reason="real data not present")


@pytest.fixture(scope="module")
def built():
    cfg = load.load_config(os.path.join(ROOT, "config.yaml"))
    cfg["paths"]["data_dir"] = os.path.join(ROOT, "data")
    data = load.load_data(cfg)
    return cfg, data, state.build_states(data["bbb"], data["matches"], cfg)


def test_counts(built):
    _, data, _ = built
    assert data["meta"]["raw_rows"] == 200999 and data["meta"]["raw_matches"] == 867
    assert data["meta"]["matches_without_ball_data"] == 16


def test_state_checks(built):
    cfg, data, df = built
    out = state.run_state_checks(df, data["matches"], cfg)   # margins, legal balls <= 120, wickets <= 10
    assert out["max_legal_balls"] <= 120 and out["max_wickets"] <= 10 and out["run_margin_matches_checked"] > 0


def test_other_competitions(built):
    _, data, _ = built
    m = data["matches"]
    assert set(m.loc[m.comp_type == "OTHER", "comp_id"]) <= {2670, 3289}
