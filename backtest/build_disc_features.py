"""Disclosure-activity features per (inn, as_of), from messages dated on or before as_of."""
import json
import os
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
from panel_config import AS_OF, SUFFIX


def d(s):
    try:
        dd, mm, yy = s.split(".")
        return date(int(yy), int(mm), int(dd))
    except Exception:  # noqa: BLE001
        return None


msgs = defaultdict(list)
seen = set()
with open(os.path.join(BASE, "prime_messages.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        key = (r["inn"], r["guid"], r["date_feed"], r["title"])
        if key in seen:
            continue
        seen.add(key)
        dt = d(r["date_feed"]) or d(r["date_site"])
        if dt:
            msgs[r["inn"]].append((dt, r["title"].lower()))

events = json.load(open(os.path.join(BASE, "events.json"), encoding="utf-8"))
ev_by_inn = defaultdict(list)
for e in events:
    ev_by_inn[e["inn"]].append((date.fromisoformat(e["date"]), e["cls"]))

rows = []
for inn, items in msgs.items():
    items.sort()
    for as_of in AS_OF:
        past = [x for x in items if x[0] <= as_of]
        if not past:
            continue
        y1 = as_of - timedelta(days=365)
        last12 = [x for x in past if x[0] > y1]
        ev = [x for x in ev_by_inn.get(inn, []) if x[0] <= as_of]
        ev12 = [x for x in ev if x[0] > y1]
        rows.append(
            {
                "inn": inn,
                "as_of": as_of.isoformat(),
                "msg_total": len(past),
                "msg_12m": len(last12),
                "days_since_last_msg": (as_of - past[-1][0]).days,
                "days_since_first_msg": (as_of - past[0][0]).days,
                "coupon_12m": sum(1 for _, c in ev12 if c == "coupon_paid"),
                "rating_12m": sum(1 for _, c in ev12 if c == "rating"),
                "has_rating_ever": int(any(c == "rating" for _, c in ev)),
                "redeem_12m": sum(1 for _, c in ev12 if c == "redeem"),
                "delist_12m": sum(1 for _, c in ev12 if c == "delist"),
                "accel_prior": sum(1 for _, c in ev if c == "accel"),
                "suspend_prior": sum(1 for _, c in ev if c == "suspend"),
                "default_prior": sum(1 for _, c in ev if c == "default"),
                "material_12m": sum(
                    1 for dt, t in last12 if "существенное влияние" in t
                ),
                "observable": int(len(last12) > 0),
                "active_bond": int(
                    any(c in ("coupon_paid", "redeem") for _, c in ev12)
                ),
            }
        )

df = pd.DataFrame(rows)
print("rows:", len(df), "issuers:", df["inn"].nunique())
print(df.groupby("as_of")[["observable", "active_bond"]].sum())
df.to_csv(os.path.join(BASE, "disc_features%s.csv" % SUFFIX), index=False, encoding="utf-8")
print("saved disc_features.csv")
