"""Discrete-time hazard panel — the honest way to get more training signal.

Each (issuer, month) is one observation and the training label is "an event happens
in THIS month". That is what stops a single default from being copied into eight
overlapping positive rows, which is what a long window on a monthly grid would do.

Evaluation stays comparable to the semi-annual run: rank issuers at 31.12.2025 by
the fitted monthly hazard and score that ranking against the 250-day outcome.
Confidence intervals are bootstrapped over ISSUERS, not rows — rows of the same
issuer are the same bet made 36 times.
"""
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import model as M

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HARD = "--hard" in sys.argv
STRESS = {"default"} if HARD else {"default", "accel", "bankruptcy", "suspend",
                                   "restructuring", "liquidation"}
TEST_AS_OF = "2025-12-31"
EVAL_HORIZON = 250
TRAIN_CUT = (date.fromisoformat(TEST_AS_OF) - timedelta(days=EVAL_HORIZON)).isoformat()
FEATS = M.FIN + M.DISC


def load_panel():
    fin = pd.read_csv(os.path.join(BASE, "features_monthly.csv"), dtype={"inn": str})
    disc = pd.read_csv(os.path.join(BASE, "disc_features_monthly.csv"), dtype={"inn": str})
    df = fin.merge(disc, on=["inn", "as_of"], how="inner")
    mp = os.path.join(BASE, "market_monthly.csv")
    if os.path.exists(mp):
        df = df.merge(pd.read_csv(mp, dtype={"inn": str}), on=["inn", "as_of"], how="left")
        df["log_trades"] = np.log1p(df["trades"].fillna(0))
    df["log_assets"] = np.log1p(df["assets"].clip(lower=0))
    yr = df["as_of"].str[:4].astype(int)
    df["missed_filing"] = (df["period"] < yr - 1).astype(int)

    ev = json.load(open(os.path.join(BASE, "events.json"), encoding="utf-8"))
    by_inn = {}
    for e in ev:
        if e["cls"] in STRESS:
            by_inn.setdefault(e["inn"], []).append(date.fromisoformat(e["date"]))

    y1, y250 = [], []
    for inn, as_of in zip(df["inn"].values, df["as_of"].values):
        t = date.fromisoformat(as_of)
        ds = by_inn.get(inn, [])
        y1.append(int(any(t < d <= t + timedelta(days=31) for d in ds)))
        y250.append(int(any(t < d <= t + timedelta(days=EVAL_HORIZON) for d in ds)))
    df["y_hazard"] = y1
    df["y"] = y250

    nm = (df["full_name"].fillna("") + " " + df["name"].fillna("")).str.lower()
    df["is_spv"] = nm.apply(lambda s: int(any(p in s for p in M.SPV_PAT)))
    df["is_fin"] = df["okved"].fillna("").astype(str).str.startswith(("64", "65", "66")).astype(int)
    return df


def main():
    df = load_panel()
    u = M.universe(df)
    print(f"panel rows={len(u)} issuers={u['inn'].nunique()} months={u['as_of'].nunique()}")

    train = u[u["as_of"] <= TRAIN_CUT]
    test = u[u["as_of"] == TEST_AS_OF]
    print(f"train  rows={len(train)} issuers={train['inn'].nunique()} "
          f"month-events={int(train['y_hazard'].sum())} "
          f"(distinct issuers with an event: "
          f"{train[train.y_hazard == 1]['inn'].nunique()})")
    print(f"test   rows={len(test)} events(250d)={int(test['y'].sum())} "
          f"base={test['y'].mean():.1%}\n")

    for tag, feats in (("fin + disclosure", FEATS),
                       ("fin + disc + market", FEATS + M.MKT)):
        feats = [c for c in feats if c in train.columns]
        Xtr, Xte, names = M.prep(train, test, feats)
        # scale the ridge with sample size so a 7x bigger panel is not 7x less regularised
        w = M.fit_logit(Xtr, train["y_hazard"].values.astype(float),
                        l2=5.0 * len(train) / 913)
        s = Xte @ w
        lo, med, hi = M.boot_auc(test["y"].values, s, test["inn"].values)
        print(f"[{tag}]  hazard-trained, evaluated on the 250-day outcome")
        M_report(test["y"].values, s)
        print(f"      bootstrap over issuers: AUC {med:.3f}  90% CI [{lo:.3f}, {hi:.3f}]\n")
        if tag == "fin + disclosure":
            t = test.copy()
            t["score"] = s
            t.sort_values("score", ascending=False)[
                ["inn", "score", "y", "equity_ratio", "icr", "roa",
                 "debt_to_assets", "period"]].head(30).to_csv(
                os.path.join(BASE, "panel_top.csv"), index=False, encoding="utf-8")

    # what the semi-annual design gave, for reference
    print("[reference] semi-annual design, same test date, fin + disclosure: AUC 0.755")


def M_report(y, s):
    y = np.asarray(y)
    base = y.mean()
    print(f"      n={len(y)} events={int(y.sum())} base={base:.1%} "
          f"AUC={M.auc(y, s):.3f}")
    order = np.argsort(-np.asarray(s))
    for frac in (0.10, 0.20):
        k = max(1, int(len(y) * frac))
        cap = int(y[order[:k]].sum())
        print(f"      top{int(frac*100):>3}% ({k:3}): {cap}/{int(y.sum())}, "
              f"precision {cap/k:.1%}, lift {(cap/k)/base:.1f}x")


if __name__ == "__main__":
    main()
