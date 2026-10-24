"""Out-of-time default backtest: features as of T, label is a stress event in (T, T + horizon]."""
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "data")
DATASET = os.path.join(HERE, "dataset", "semiannual.csv.gz")
HORIZON = 250  # days, same for every fold
HARD_ONLY = "--hard" in sys.argv
STRESS_ALL = {"default", "accel", "bankruptcy", "suspend", "restructuring", "liquidation"}
STRESS = {"default"} if HARD_ONLY else STRESS_ALL

FIN = ["equity_ratio", "debt_to_assets", "icr", "current_ratio", "quick_ratio",
       "cash_to_assets", "recv_share", "short_debt_share", "roa", "op_margin",
       "rev_growth", "ocf_to_debt", "neg_equity", "net_loss", "ocf_negative",
       "has_audit", "pub_lag_days", "log_assets", "stale_days", "missed_filing"]
DISC = ["msg_12m", "days_since_last_msg", "rating_12m", "has_rating_ever",
        "material_12m", "accel_prior", "suspend_prior"]
SMALL = ["equity_ratio", "icr", "debt_to_assets", "roa", "cash_to_assets",
         "has_rating_ever"]
MKT = ["min_price", "med_price", "max_yield", "med_yield", "med_duration",
       "n_quoted_bonds", "log_trades"]

SPV_PAT = ("ипотечный агент", "специализированное финансовое", "сфо ", "ипотечная")


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_auc(y, s, groups, n=400, seed=1):
    """Percentile bootstrap over issuers: rows of one issuer are resampled together."""
    rng = np.random.default_rng(seed)
    # first-appearance order, so the draw is the same for INNs and hashes
    uniq = pd.unique(np.asarray(groups))
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        a = auc(np.asarray(y)[idx], np.asarray(s)[idx])
        if not np.isnan(a):
            out.append(a)
    return np.percentile(out, [5, 50, 95])


def boot_gain(y, s, s0, groups, n=1000, seed=1):
    """Paired bootstrap over issuers of AUC(s) - AUC(s0): 90% CI and share of draws <= 0."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = pd.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    y, s, s0 = np.asarray(y), np.asarray(s), np.asarray(s0)
    out = []
    for _ in range(n):
        idx = np.concatenate([rows[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        d = auc(y[idx], s[idx]) - auc(y[idx], s0[idx])
        if not np.isnan(d):
            out.append(d)
    out = np.asarray(out)
    lo, hi = np.percentile(out, [5, 95])
    return lo, hi, float((out <= 0).mean())


def has_event(dates, t, horizon=HORIZON):
    """Label: an event strictly after the snapshot date and no later than t + horizon."""
    end = t + timedelta(days=horizon)
    return int(any(t < d <= end for d in dates))


def load():
    if not os.path.exists(os.path.join(BASE, "features.csv")):
        return load_dataset()
    fin = pd.read_csv(os.path.join(BASE, "features.csv"), dtype={"inn": str})
    disc = pd.read_csv(os.path.join(BASE, "disc_features.csv"), dtype={"inn": str})
    df = fin.merge(disc, on=["inn", "as_of"], how="inner")
    mkt_path = os.path.join(BASE, "market.csv")
    if os.path.exists(mkt_path):
        mkt = pd.read_csv(mkt_path, dtype={"inn": str})
        df = df.merge(mkt, on=["inn", "as_of"], how="left")
        df["log_trades"] = np.log1p(df["trades"].fillna(0))
    df["log_assets"] = np.log1p(df["assets"].clip(lower=0))
    as_of_year = df["as_of"].str[:4].astype(int)
    df["missed_filing"] = (df["period"] < as_of_year - 1).astype(int)

    ev = json.load(open(os.path.join(BASE, "events.json"), encoding="utf-8"))
    by_inn = {}
    for e in ev:
        if e["cls"] in STRESS:
            by_inn.setdefault(e["inn"], []).append(date.fromisoformat(e["date"]))

    df["y"] = [has_event(by_inn.get(inn, []), date.fromisoformat(t))
               for inn, t in zip(df["inn"].values, df["as_of"].values)]

    name = (df["full_name"].fillna("") + " " + df["name"].fillna("")).str.lower()
    df["is_spv"] = name.apply(lambda s: int(any(p in s for p in SPV_PAT)))
    okved = df["okved"].fillna("").astype(str)
    df["is_fin"] = okved.str.startswith(("64", "65", "66")).astype(int)
    return df


def load_dataset():
    """The published table: same rows and features as load(), INN replaced by a keyed hash."""
    df = pd.read_csv(DATASET, dtype={"issuer_id": str}, float_precision="round_trip")
    df["y"] = df["y_default"] if STRESS == {"default"} else df["y_stress"]
    # "inn" holds the hash here
    return df.rename(columns={"issuer_id": "inn"})


def universe(df):
    keep = (df["observable"] == 1) & (df["is_spv"] == 0) & (df["is_fin"] == 0) & \
           (df["is_cb"] == 0) & (df["default_prior"] == 0)
    return df[keep].copy()


def split(u, horizon=HORIZON):
    """Train on snapshots whose label window closes before the test snapshot."""
    test_t = sorted(u["as_of"].unique())[-1]
    # drop snapshots whose label window overlaps the test window
    cut = (date.fromisoformat(test_t) - timedelta(days=horizon)).isoformat()
    return u[u["as_of"] <= cut], u[u["as_of"] == test_t]


def prep(train, test, feats):
    med = train[feats].median()
    lo, hi = train[feats].quantile(0.02), train[feats].quantile(0.98)

    def mat(d):
        x = d[feats].astype(float).clip(lower=lo, upper=hi, axis=1)
        return x.fillna(med), x.isna().astype(float)

    xtr, mtr = mat(train)
    xte, mte = mat(test)
    mu, sd = xtr.mean(), xtr.std().replace(0, 1)
    keep = [c for c in feats if mtr[c].mean() < 0.5 and sd[c] > 0]
    miss = [c for c in keep if 0.03 < mtr[c].mean() < 0.5]
    Xtr = np.hstack([((xtr - mu) / sd)[keep].values, mtr[miss].values,
                     np.ones((len(xtr), 1))])
    Xte = np.hstack([((xte - mu) / sd)[keep].values, mte[miss].values,
                     np.ones((len(xte), 1))])
    return Xtr, Xte, keep + [c + "__miss" for c in miss] + ["const"]


def fit_logit(X, y, l2=5.0):
    w = np.zeros(X.shape[1])
    R = np.eye(X.shape[1]) * l2
    R[-1, -1] = 0.0
    for _ in range(300):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        g = X.T @ (y - p) - R @ w
        H = X.T @ (X * W[:, None]) + R
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def report(y, s, label, groups=None):
    y = np.asarray(y)
    base = y.mean() if len(y) else 0
    print(f"  {label:26} n={len(y):4} events={int(y.sum()):3} base={base:5.1%} "
          f"AUC={auc(y, s):.3f}")
    order = np.argsort(-np.asarray(s))
    for frac in (0.10, 0.20):
        k = max(1, int(len(y) * frac))
        cap = int(y[order[:k]].sum())
        lift = (cap / k) / base if base else 0
        print(f"      top{int(frac*100):>3}% ({k:3} names): catches {cap}/{int(y.sum())} "
              f"— precision {cap/k:5.1%}, lift {lift:.1f}x")
    if groups is not None:
        lo, _, hi = boot_auc(y, s, groups)
        print(f"      bootstrap over issuers: AUC 90% CI [{lo:.3f}, {hi:.3f}]")


def main():
    df = load()
    print(f"merged rows={len(df)} issuers={df['inn'].nunique()}  "
          f"label={'default only' if HARD_ONLY else 'stress'} horizon={HORIZON}d")
    u = universe(df)
    print(f"in universe rows={len(u)} issuers={u['inn'].nunique()} "
          f"(dropped {len(df)-len(u)}: spv/fin/cb/unobservable/already-defaulted)")
    for t, g in u.groupby("as_of"):
        print(f"   as_of {t}: n={len(g):4} events={int(g['y'].sum()):3} "
              f"({g['y'].mean():.1%})")

    train, test = split(u)
    test_t = test["as_of"].iloc[0]
    print(f"\ntrain={sorted(train['as_of'].unique())} n={len(train)} "
          f"events={int(train['y'].sum())}")
    print(f"test ={test_t} n={len(test)} events={int(test['y'].sum())}\n")

    scores = {}
    for tag, feats in (("size only (null model)", ["log_assets"]),
                       ("financials only", FIN),
                       ("fin + disclosure", FIN + DISC),
                       ("6-feature scorecard", SMALL),
                       ("market only", MKT),
                       ("fin + disclosure + market", FIN + DISC + MKT)):
        Xtr, Xte, names = prep(train, test, feats)
        w = fit_logit(Xtr, train["y"].values.astype(float))
        scores[tag] = Xte @ w
        print(f"[{tag}]")
        report(train["y"].values, Xtr @ w, "in-sample")
        report(test["y"].values, Xte @ w, "OUT-OF-TIME", test["inn"].values)
        if tag == "fin + disclosure":
            print("   weights:", ", ".join(
                f"{n}={v:+.2f}" for n, v in
                sorted(zip(names, w), key=lambda x: -abs(x[1]))[:10] if n != "const"))
            t2 = test.copy()
            t2["score"] = Xte @ w
            t2["pd"] = 1 / (1 + np.exp(-t2["score"]))
            os.makedirs(BASE, exist_ok=True)
            t2.sort_values("score", ascending=False)[
                ["inn", "pd", "y", "equity_ratio", "icr", "roa",
                 "debt_to_assets", "period"]
            ].head(30).to_csv(os.path.join(BASE, "top_risk.csv"), index=False,
                              encoding="utf-8")
        print()

    # gain over the size-only model, paired by issuer
    print("[paired bootstrap over issuers: AUC gain over the size-only model]")
    null = scores["size only (null model)"]
    for tag in ("financials only", "fin + disclosure", "fin + disclosure + market"):
        lo, hi, p = boot_gain(test["y"].values, scores[tag], null, test["inn"].values)
        gain = auc(test["y"].values, scores[tag]) - auc(test["y"].values, null)
        print(f"   {tag:26} {gain:+.3f}  90% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"share of draws <= 0: {p:.1%}")
    print()

    print("[in-period 5-fold CV on the test period — is there ANY learnable signal?]")
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(test))
    folds = np.array_split(idx, 5)
    for tag, feats in (("fin + disclosure", FIN + DISC),
                       ("fin + disc + market", FIN + DISC + MKT)):
        oof = np.zeros(len(test))
        for f in folds:
            mask = np.ones(len(test), bool)
            mask[f] = False
            tr, te = test[mask], test[~mask]
            if tr["y"].sum() < 3:
                continue
            Xa, Xb, _ = prep(tr, te, feats)
            wcv = fit_logit(Xa, tr["y"].values.astype(float))
            oof[~mask] = Xb @ wcv
        print(f"   {tag:22} out-of-fold AUC={auc(test['y'].values, oof):.3f}")
    print()

    print("[single-feature baselines, out-of-time]")
    for c, sgn in (("icr", -1), ("equity_ratio", -1), ("roa", -1),
                   ("current_ratio", -1), ("debt_to_assets", 1),
                   ("has_rating_ever", -1), ("log_assets", -1),
                   ("max_yield", 1), ("min_price", -1), ("med_yield", 1),
                   ("accel_prior", 1)):
        if c not in test.columns:
            continue
        v = test[c].astype(float)
        v = v.fillna(v.median())
        print(f"   {c:18} AUC={auc(test['y'].values, sgn * v.values):.3f}")


if __name__ == "__main__":
    main()
