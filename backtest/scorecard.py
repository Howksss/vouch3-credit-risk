"""WOE scorecard: bin edges fixed by credit logic, only bin weights are estimated."""
import json
import os
import sys

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HORIZON = 250
HARD_ONLY = "--hard" in sys.argv
STRESS = {"default"} if HARD_ONLY else {
    "default", "accel", "bankruptcy", "suspend", "restructuring", "liquidation"}
PRIOR = 8.0  # shrinkage toward the base rate

# (feature, edges, higher_is_worse)
BLOCKS = [
    ("equity_ratio", [0.0, 0.10, 0.25, 0.50], False),
    ("debt_to_assets", [0.10, 0.30, 0.50, 0.70], True),
    ("icr", [1.0, 2.0, 4.0, 8.0], False),
    ("current_ratio", [1.0, 1.5, 2.0], False),
    ("roa", [0.0, 0.03, 0.08], False),
    ("cash_to_assets", [0.01, 0.05], False),
    ("ocf_to_debt", [0.0, 0.10], False),
    ("short_debt_share", [0.30, 0.70], True),
    ("log_assets", [19.0, 20.5, 22.0, 23.5], False),
    ("rev_growth", [0.9, 1.05, 1.4], None),
    ("accel_prior", [0.5], True),
    ("has_rating_ever", [0.5], False),
    ("missed_filing", [0.5], True),
]


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s)
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def binize(v, edges):
    """Bin index; last index is reserved for missing."""
    out = np.full(len(v), len(edges) + 1, dtype=int)
    ok = v.notna().values
    out[ok] = np.digitize(v[ok].values.astype(float), edges)
    return out


def fit_woe(train, test):
    base = train["y"].mean()
    log_odds_base = np.log(base / (1 - base))
    card = {}
    s_tr = np.full(len(train), 0.0)
    s_te = np.full(len(test), 0.0)
    for feat, edges, _worse in BLOCKS:
        btr = binize(train[feat], edges)
        bte = binize(test[feat], edges)
        w = {}
        for b in sorted(set(btr)):
            m = btr == b
            n, k = m.sum(), train["y"].values[m].sum()
            p = (k + PRIOR * base) / (n + PRIOR)
            w[b] = np.log(p / (1 - p)) - log_odds_base
        card[feat] = {"edges": edges, "weights": {int(k): round(v, 3) for k, v in w.items()},
                      "counts": {int(b): int((btr == b).sum()) for b in sorted(set(btr))},
                      "events": {int(b): int(train["y"].values[btr == b].sum())
                                 for b in sorted(set(btr))}}
        s_tr += np.array([w.get(b, 0.0) for b in btr])
        s_te += np.array([w.get(b, 0.0) for b in bte])
    return card, s_tr + log_odds_base, s_te + log_odds_base


def report(y, s, label):
    y = np.asarray(y)
    base = y.mean() if len(y) else 0
    print(f"  {label:24} n={len(y):4} events={int(y.sum()):3} base={base:5.1%} "
          f"AUC={auc(y, s):.3f}")
    order = np.argsort(-np.asarray(s))
    for frac in (0.10, 0.20):
        k = max(1, int(len(y) * frac))
        cap = int(y[order[:k]].sum())
        print(f"      top{int(frac*100):>3}% ({k:3}): {cap}/{int(y.sum())} events, "
              f"precision {cap/k:5.1%}, lift {(cap/k)/base if base else 0:.1f}x")


def main():
    import model as M

    M.STRESS = STRESS
    df = M.load()
    u = M.universe(df)
    train, test = M.split(u, HORIZON)
    test_t = test["as_of"].iloc[0]
    print(f"train n={len(train)} events={int(train['y'].sum())} | "
          f"test {test_t} n={len(test)} events={int(test['y'].sum())}")

    card, s_tr, s_te = fit_woe(train, test)
    print("\n[WOE scorecard]")
    report(train["y"].values, s_tr, "in-sample")
    report(test["y"].values, s_te, "OUT-OF-TIME")

    print("\n[bin weights: + means riskier]")
    for feat, edges, _ in BLOCKS:
        c = card[feat]
        lbl = []
        for b in sorted(c["weights"]):
            rng = ("<%g" % edges[0] if b == 0 else
                   (">=%g" % edges[-1] if b == len(edges) else
                    ("miss" if b == len(edges) + 1 else
                     "%g-%g" % (edges[b - 1], edges[b]))))
            lbl.append(f"{rng}:{c['weights'][b]:+.2f}(n={c['counts'][b]},e={c['events'][b]})")
        print(f"  {feat:18} " + "  ".join(lbl))

    t = test.copy()
    t["score"] = s_te
    t["pd"] = 1 / (1 + np.exp(-s_te))
    os.makedirs(BASE, exist_ok=True)
    cols = ["inn", "pd", "y", "equity_ratio", "icr", "roa", "debt_to_assets",
            "current_ratio", "log_assets", "period"]
    t.sort_values("score", ascending=False)[cols].head(30).to_csv(
        os.path.join(BASE, "scorecard_top.csv"), index=False, encoding="utf-8")
    json.dump(card, open(os.path.join(BASE, "scorecard.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nsaved scorecard_top.csv, scorecard.json")


if __name__ == "__main__":
    main()
