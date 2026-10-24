"""Does the panel help at all? Sweep the hazard window and the ridge strength.

The panel adds rows, not events — the same ~33 issuers with events are simply cut
into months. This sweep is the check on whether that re-slicing buys anything.
"""
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

import model as M
import panel as P

BASE = P.BASE

df = P.load_panel()
u = M.universe(df)
ev = json.load(open(os.path.join(BASE, "events.json"), encoding="utf-8"))
by_inn = {}
for e in ev:
    if e["cls"] in P.STRESS:
        by_inn.setdefault(e["inn"], []).append(date.fromisoformat(e["date"]))


def label(sub, days):
    out = []
    for inn, as_of in zip(sub["inn"].values, sub["as_of"].values):
        t = date.fromisoformat(as_of)
        out.append(int(any(t < d <= t + timedelta(days=days) for d in by_inn.get(inn, []))))
    return np.array(out, float)


test = u[u["as_of"] == P.TEST_AS_OF]
feats = [c for c in M.FIN + M.DISC if c in u.columns]
rows = []
for win in (31, 90, 180, 250):
    cut = (date.fromisoformat(P.TEST_AS_OF) - timedelta(days=max(win, 250))).isoformat()
    for step in (1, 3, 6):
        months = sorted(u[u["as_of"] <= cut]["as_of"].unique())[::step]
        train = u[u["as_of"].isin(months)]
        ytr = label(train, win)
        if ytr.sum() < 5:
            continue
        for l2 in (2.0, 5.0, 20.0, 60.0):
            Xtr, Xte, _ = M.prep(train, test, feats)
            w = M.fit_logit(Xtr, ytr, l2=l2)
            a = M.auc(test["y"].values, Xte @ w)
            rows.append({"window_d": win, "month_step": step, "l2": l2,
                         "train_rows": len(train), "train_events": int(ytr.sum()),
                         "issuers_with_event": int(train.loc[ytr == 1, "inn"].nunique()),
                         "test_AUC": round(a, 3)})

r = pd.DataFrame(rows).sort_values("test_AUC", ascending=False)
pd.set_option("display.width", 200)
print(r.head(15).to_string(index=False))
print("\nworst:")
print(r.tail(5).to_string(index=False))
print("\nsemi-annual reference (6 snapshots, 55 events, l2=5): 0.755")
r.to_csv(os.path.join(BASE, "panel_sweep.csv"), index=False)
