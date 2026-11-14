"""Phase 2: baseline expectation models and win probability.

Everything is split by match_id. Model *selection* uses a held-out set of matches;
the predictions used downstream are cross-fitted (each match is scored by models
that never saw that match).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6
OUTCOME_LABELS = ["dot", "1", "2", "4", "6", "wicket"]
OUTCOME_RUNS = np.array([0, 1, 2, 4, 6, 0])
STATE_FEATURES = ["innings", "balls_remaining", "wickets_before", "rrr", "comp_code", "season"]


def outcome_class(df: pd.DataFrame) -> np.ndarray:
    r = df["team_runs"].to_numpy()
    c = np.select([r <= 0, r == 1, r <= 3, r <= 5], [0, 1, 2, 3], default=4)
    return np.where(df["is_wicket"].to_numpy(), 5, c)


def split_matches(match_ids, test_fraction: float, seed: int):
    ids = np.array(sorted(set(match_ids)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    n_test = max(1, int(round(len(ids) * test_fraction)))
    return set(perm[n_test:]), set(perm[:n_test])


def assign_folds(match_ids, n_folds: int, seed: int) -> dict:
    ids = np.array(sorted(set(match_ids)))
    perm = np.random.default_rng(seed + 1).permutation(ids)
    return {m: i % n_folds for i, m in enumerate(perm)}


# ----------------------------------------------------------------- lookup table
class SmoothedLookup:
    """Hierarchically smoothed state table.

    Cells: innings x balls-remaining bin x wickets x required-rate bin x competition.
    Each cell mean is shrunk toward its parent (same cell with the last key dropped).
    """
    BASES = np.array([3, 64, 16, 16, 4])
    LEVELS = [5, 4, 3, 2, 0]  # number of leading keys kept

    def __init__(self, cfg: dict):
        lk = cfg["models"]["lookup"]
        self.k = lk["prior_strength"]
        self.bin = lk["balls_bin"]
        self.wcap = lk["wickets_cap"]
        self.edges = np.array(lk["rrr_edges"], dtype=float)

    def _codes(self, df):
        inn = df["innings"].to_numpy().astype(np.int64)
        bb = np.clip((df["balls_remaining"].to_numpy() - 1) // self.bin, 0, 63).astype(np.int64)
        wk = np.minimum(df["wickets_before"].to_numpy(), self.wcap).astype(np.int64)
        rrr = df["rrr"].to_numpy(dtype=float)
        rb = np.where(np.isnan(rrr), 0, 1 + np.searchsorted(self.edges, np.nan_to_num(rrr))).astype(np.int64)
        comp = df["comp_code"].to_numpy().astype(np.int64)
        keys = [inn, bb, wk, rb, comp]
        out = []
        for keep in self.LEVELS:
            code = np.zeros(len(df), dtype=np.int64)
            for j in range(keep):
                code = code * self.BASES[j] + keys[j]
            out.append(code)
        return out

    def fit(self, df, Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.dim = Y.shape[1]
        codes = self._codes(df)
        self.tables = [None] * len(self.LEVELS)
        self.global_ = Y.mean(axis=0)
        cols = list(range(self.dim))
        parent_tab = None
        for li in range(len(self.LEVELS) - 1, -1, -1):
            fr = pd.DataFrame(Y, columns=cols)
            fr["code"] = codes[li]
            fr["n"] = 1.0
            if li < len(self.LEVELS) - 1:
                fr["parent"] = codes[li + 1]
            g = fr.groupby("code")
            sums, n = g[cols].sum(), g["n"].sum()
            if parent_tab is None:
                prior = np.tile(self.global_, (len(sums), 1))
            else:
                prior = parent_tab.reindex(g["parent"].first().to_numpy()).to_numpy()
            est = (sums.to_numpy() + self.k * prior) / (n.to_numpy()[:, None] + self.k)
            parent_tab = pd.DataFrame(est, index=sums.index, columns=cols)
            self.tables[li] = parent_tab
        return self

    def predict(self, df):
        codes = self._codes(df)
        out = np.full((len(df), self.dim), np.nan)
        for li in range(len(self.LEVELS)):
            miss = np.isnan(out[:, 0])
            if not miss.any():
                break
            out[miss] = self.tables[li].reindex(codes[li][miss]).to_numpy()
        miss = np.isnan(out[:, 0])
        out[miss] = self.global_
        return out[:, 0] if self.dim == 1 else out


# ----------------------------------------------------------------- baselines
class Baseline:
    """kind: 'reg' | 'bin' | 'multi' ; method: 'gb' | 'lookup'."""

    def __init__(self, kind: str, method: str, cfg: dict):
        self.kind, self.method, self.cfg = kind, method, cfg

    def fit(self, df, y):
        seed = self.cfg["seed"]
        if self.method == "lookup":
            Y = np.eye(len(OUTCOME_LABELS))[y] if self.kind == "multi" else np.asarray(y, dtype=float)
            self.m = SmoothedLookup(self.cfg).fit(df, Y)
            return self
        p = dict(self.cfg["models"]["gb"])
        X = df[STATE_FEATURES].to_numpy(dtype=float)
        cls = HistGradientBoostingRegressor if self.kind == "reg" else HistGradientBoostingClassifier
        self.m = cls(categorical_features=[4], random_state=seed, **p).fit(X, y)
        return self

    def predict(self, df):
        if self.method == "lookup":
            out = self.m.predict(df)
            if self.kind == "multi":
                out = np.clip(out, EPS, None)
                out = out / out.sum(axis=1, keepdims=True)
            elif self.kind == "bin":
                out = np.clip(out, EPS, 1 - EPS)
            return out
        X = df[STATE_FEATURES].to_numpy(dtype=float)
        if self.kind == "reg":
            return self.m.predict(X)
        pr = self.m.predict_proba(X)
        if self.kind == "bin":
            return np.clip(pr[:, 1], EPS, 1 - EPS)
        full = np.full((len(df), len(OUTCOME_LABELS)), EPS)
        full[:, self.m.classes_.astype(int)] = pr
        return full / full.sum(axis=1, keepdims=True)


def _loss(kind, y, p):
    if kind == "reg":
        return float(np.mean((y - p) ** 2))
    if kind == "bin":
        p = np.clip(p, EPS, 1 - EPS)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))))


def baseline_targets(df):
    """name -> (kind, row mask, target). xr_bat is defined on balls faced only."""
    ok = df["state_valid"].to_numpy()
    return {
        "xr_bat": ("reg", ok & df["faced"].to_numpy(), df["runs"].to_numpy(dtype=float)),
        "xr_bowl": ("reg", ok, df["bowler_runs"].to_numpy(dtype=float)),
        "p_dismissal": ("bin", ok, df["is_wicket"].to_numpy().astype(int)),
        "p_bowler_wicket": ("bin", ok, df["bowler_wicket"].to_numpy().astype(int)),
        "outcome": ("multi", ok, outcome_class(df)),
    }


# ----------------------------------------------------------------- win probability
def _logit_feats(kind, a, balls, wk, comp):
    b = balls / 120.0
    if kind == 2:
        rrr = np.clip(a * 6.0 / np.maximum(balls, 1), 0, 40)
        f = [rrr, wk, b, rrr * wk, np.log1p(np.maximum(a, 0)), rrr * b, wk * b]
    else:
        elapsed = np.maximum(120 - balls, 0)
        rate = a / (elapsed + 12.0)
        f = [a / 100.0, b, wk, rate, wk * b, rate * b]
    f += [(comp == 1).astype(float), (comp == 2).astype(float)]
    return np.column_stack(f)


class WPModel:
    """Win probability of the batting side. Innings 1 and 2 are separate models."""

    def __init__(self, cfg, method="gb"):
        self.cfg, self.method = cfg, method

    def _fit_one(self, kind, a, balls, wk, comp, season, y):
        # soft labels (ties = 0.5) -> duplicate rows with weights
        y1 = np.concatenate([np.ones(len(y)), np.zeros(len(y))])
        w = np.concatenate([y, 1 - y])
        keep = w > 0
        rep = lambda v: np.concatenate([v, v])[keep]
        a, balls, wk, comp, season = map(rep, (a, balls, wk, comp, season))
        y1, w = y1[keep], w[keep]
        if self.method == "gb":
            X = np.column_stack([a, balls, wk, comp, season]).astype(float)
            mono = [1, 1, -1, 0, 0] if kind == 1 else [-1, 1, -1, 0, 0]
            m = HistGradientBoostingClassifier(monotonic_cst=mono, categorical_features=[3],
                                               random_state=self.cfg["seed"], **self.cfg["models"]["gb_wp"])
            return m.fit(X, y1, sample_weight=w)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        return m.fit(_logit_feats(kind, a, balls, wk, comp), y1, logisticregression__sample_weight=w)

    def fit(self, df):
        d = df[df["wp_trainable"] & df["state_valid"]]
        d1 = d[d["innings"] == 1]
        d2 = d[(d["innings"] == 2) & (d["runs_needed"] > 0)]
        g = lambda x, c: x[c].to_numpy(dtype=float)
        self.m1 = self._fit_one(1, g(d1, "score_before"), g(d1, "balls_remaining"), g(d1, "wickets_before"),
                                g(d1, "comp_code"), g(d1, "season"), g(d1, "bat_win"))
        self.m2 = self._fit_one(2, g(d2, "runs_needed"), g(d2, "balls_remaining"), g(d2, "wickets_before"),
                                g(d2, "comp_code"), g(d2, "season"), g(d2, "bat_win"))
        return self

    def _raw(self, kind, a, balls, wk, comp, season):
        if len(a) == 0:
            return np.zeros(0)
        m = self.m1 if kind == 1 else self.m2
        if self.method == "gb":
            X = np.column_stack([a, balls, wk, comp, season]).astype(float)
        else:
            X = _logit_feats(kind, a, balls, wk, comp)
        return np.clip(m.predict_proba(X)[:, 1], EPS, 1 - EPS)

    def wp(self, innings, score, balls, wk, target, len2, comp, season):
        """WP of the side batting in `innings` at an arbitrary (possibly terminal) state."""
        innings, score, balls, wk, target, len2, comp, season = [
            np.asarray(v, dtype=float) for v in (innings, score, balls, wk, target, len2, comp, season)]
        out = np.full(len(innings), np.nan)
        over = (balls <= 0) | (wk >= 10)
        i2 = innings == 2
        needed = target - score
        won = i2 & (needed <= 0)
        out[won] = 1.0
        lost = i2 & ~won & over
        out[lost] = np.where(needed[lost] == 1, 0.5, 0.0)
        live2 = i2 & ~won & ~over
        out[live2] = self._raw(2, needed[live2], balls[live2], wk[live2], comp[live2], season[live2])
        i1 = ~i2
        end1 = i1 & over  # innings break: the chase starts needing score + 1
        out[end1] = 1 - self._raw(2, score[end1] + 1, len2[end1], np.zeros(end1.sum()), comp[end1], season[end1])
        live1 = i1 & ~over
        out[live1] = self._raw(1, score[live1], balls[live1], wk[live1], comp[live1], season[live1])
        return out

    def wp_state(self, d):
        return self.wp(d["innings"], d["score_before"], d["balls_remaining"], d["wickets_before"],
                       d["target"].fillna(0), d["len2"], d["comp_code"], d["season"])


def _soft_logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(y, p, bins):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    fr = pd.DataFrame({"bin": idx, "p": p, "y": y}).groupby("bin").agg(
        predicted=("p", "mean"), observed=("y", "mean"), balls=("p", "size")).reset_index()
    return fr


# ----------------------------------------------------------------- orchestration
def fit_all(df: pd.DataFrame, cfg: dict) -> dict:
    seed = cfg["seed"]
    train_ids, test_ids = split_matches(df["match_id"].unique(), cfg["models"]["test_fraction"], seed)
    is_tr = df["match_id"].isin(train_ids).to_numpy()
    targets = baseline_targets(df)

    # --- model comparison on held-out matches
    rows, chosen = [], {}
    for name, (kind, mask, y) in targets.items():
        tr, te = mask & is_tr, mask & ~is_tr
        losses = {}
        for method in ["gb", "lookup"]:
            m = Baseline(kind, method, cfg).fit(df[tr], y[tr])
            losses[method] = _loss(kind, y[te], m.predict(df[te]))
        if kind == "reg":
            naive = float(np.mean((y[te] - y[tr].mean()) ** 2))
        elif kind == "bin":
            naive = _loss("bin", y[te], np.full(te.sum(), y[tr].mean()))
        else:
            pri = np.bincount(y[tr], minlength=6) / tr.sum()
            naive = _loss("multi", y[te], np.tile(pri, (te.sum(), 1)))
        chosen[name] = min(losses, key=losses.get)
        rows.append({"model": name, "metric": "MSE" if kind == "reg" else "log loss",
                     "train_balls": int(tr.sum()), "test_balls": int(te.sum()),
                     "no_state_baseline": naive, "gradient_boosting": losses["gb"],
                     "smoothed_lookup": losses["lookup"], "kept": chosen[name]})
    comparison = pd.DataFrame(rows)

    # --- win probability on held-out matches
    te_df = df[~is_tr & df["wp_trainable"] & df["state_valid"]]
    y_te = te_df["bat_win"].to_numpy(dtype=float)
    wp_rows, wp_preds = [], {}
    for method in ["gb", "logistic"]:
        m = WPModel(cfg, method).fit(df[is_tr])
        p = m.wp_state(te_df)
        wp_preds[method] = p
        r = {"model": method, "test_matches": te_df["match_id"].nunique(), "test_balls": len(te_df),
             "log_loss_all": _soft_logloss(y_te, p), "brier_all": float(np.mean((y_te - p) ** 2))}
        for inn in (1, 2):
            s = (te_df["innings"] == inn).to_numpy()
            r[f"log_loss_inn{inn}"] = _soft_logloss(y_te[s], p[s])
        wp_rows.append(r)
    wp_rows.append({"model": "constant 0.5", "test_matches": te_df["match_id"].nunique(), "test_balls": len(te_df),
                    "log_loss_all": float(np.log(2)), "brier_all": 0.25,
                    "log_loss_inn1": float(np.log(2)), "log_loss_inn2": float(np.log(2))})
    wp_metrics = pd.DataFrame(wp_rows)
    wp_method = wp_metrics.iloc[:2].sort_values("log_loss_all").iloc[0]["model"]
    bins = cfg["models"]["calibration_bins"]
    calib = calibration_table(y_te, wp_preds[wp_method], bins).assign(innings="both")
    for inn in (1, 2):
        s = (te_df["innings"] == inn).to_numpy()
        calib = pd.concat([calib, calibration_table(y_te[s], wp_preds[wp_method][s], bins).assign(innings=str(inn))])

    # --- cross-fitted predictions for every delivery
    folds = assign_folds(df["match_id"].unique(), cfg["models"]["n_folds"], seed)
    fold = df["match_id"].map(folds).to_numpy()
    out = df.copy()
    out["fold"] = fold
    for c in ["xr_bat", "xr_bowl", "p_dismissal", "p_bowler_wicket"]:
        out[c] = np.nan
    po = np.full((len(df), 6), np.nan)
    wp_models, fold_models = {}, {}
    valid = df["state_valid"].to_numpy()
    for k in range(cfg["models"]["n_folds"]):
        tr, te = fold != k, (fold == k) & valid
        for name, (kind, mask, y) in targets.items():
            m = Baseline(kind, chosen[name], cfg).fit(df[tr & mask], y[tr & mask])
            fold_models[(k, name)] = m
            pred = m.predict(df[te])
            if name == "outcome":
                po[te] = pred
            else:
                out.loc[te, name] = pred
        wp_models[k] = WPModel(cfg, wp_method).fit(df[tr])
    for j, lab in enumerate(OUTCOME_LABELS):
        out[f"po_{lab}"] = po[:, j]
    return {"df": out, "comparison": comparison, "wp_metrics": wp_metrics, "wp_method": wp_method,
            "calibration": calib, "wp_models": wp_models, "chosen": chosen, "folds": folds, "fold_models": fold_models,
            "n_train_matches": len(train_ids), "n_test_matches": len(test_ids)}


def predict_baselines(mod: dict, sdf: pd.DataFrame) -> pd.DataFrame:
    """Cross-fitted baseline predictions for rows outside the training frame (super overs):
    each match is scored by the fold models that did not train on its regulation innings."""
    out = sdf.copy()
    names = ["xr_bat", "xr_bowl", "p_dismissal", "p_bowler_wicket"]
    for c in names:
        out[c] = np.nan
    fold = out["match_id"].map(mod["folds"]).fillna(0).astype(int).to_numpy()
    for k in np.unique(fold):
        m = fold == k
        for c in names:
            out.loc[m, c] = mod["fold_models"][(k, c)].predict(out[m])
    return out
