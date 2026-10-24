"""Build the default/stress event table from harvested 1prime disclosures."""
import json
import os
from collections import defaultdict
from datetime import date

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Event classes. Order matters: first match wins.
CLASSES = [
    ("default", ["неисполнение обязательств эмитента перед владельцами"]),
    ("accel", ["права требовать от эмитента досрочного погашения",
               "права требовать досрочного погашения"]),
    ("accel_end", ["прекращение у владельцев", "прекращении у владельцев"]),
    ("bankruptcy", ["банкрот", "несостоятельн"]),
    ("restructuring", ["реструктуризац"]),
    ("rating", ["рейтинг"]),
    ("suspend", ["приостановление эмиссии", "приостановление размещения",
                 "приостановление торгов"]),
    ("delist", ["исключение эмиссионных ценных бумаг эмитента из списка"]),
    ("coupon_paid", ["выплаченные доходы"]),
    ("redeem", ["погашение облигаций", "досрочное полное погашение"]),
]


def classify(title):
    t = title.lower()
    for name, kws in CLASSES:
        for k in kws:
            if k in t:
                return name
    return None


def d(s):
    try:
        dd, mm, yy = s.split(".")
        return date(int(yy), int(mm), int(dd))
    except Exception:  # noqa: BLE001
        return None


events = []
names = {}
with open(os.path.join(BASE, "prime_messages.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        c = classify(r["title"])
        if not c:
            continue
        dt = d(r["date_feed"]) or d(r["date_site"])
        if not dt:
            continue
        names[r["inn"]] = r["issuer"]
        events.append({"inn": r["inn"], "cls": c, "date": dt.isoformat(),
                       "title": r["title"], "guid": r["guid"]})

# dedupe: same inn+cls+date+guid
seen = set()
uniq = []
for e in events:
    k = (e["inn"], e["cls"], e["date"], e["guid"])
    if k in seen:
        continue
    seen.add(k)
    uniq.append(e)

by_cls = defaultdict(set)
for e in uniq:
    by_cls[e["cls"]].add(e["inn"])

print(f"events={len(uniq)}  issuers_seen={len(names)}")
for c, s in sorted(by_cls.items(), key=lambda x: -len(x[1])):
    n = sum(1 for e in uniq if e["cls"] == c)
    print(f"  {c:14} events={n:6} issuers={len(s)}")

# first default per issuer
first = {}
for e in uniq:
    if e["cls"] != "default":
        continue
    if e["inn"] not in first or e["date"] < first[e["inn"]]:
        first[e["inn"]] = e["date"]

print(f"\nissuers with >=1 default message: {len(first)}")
years = defaultdict(int)
for dt in first.values():
    years[dt[:4]] += 1
print("first-default by year:", dict(sorted(years.items())))

json.dump(uniq, open(os.path.join(BASE, "events.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
json.dump({"first_default": first, "names": names},
          open(os.path.join(BASE, "labels.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("saved events.json, labels.json")
