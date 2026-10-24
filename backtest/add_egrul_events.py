"""Add ЕГРЮЛ liquidation events (bo.nalog status codes) to the label set."""
import glob
import gzip
import json
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

events = json.load(open(os.path.join(BASE, "events.json"), encoding="utf-8"))
events = [e for e in events if e["cls"] != "liquidation"]
added = 0
for path in glob.glob(os.path.join(BASE, "bfo_raw", "*.json.gz")):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rec = json.load(fh)
    org = rec.get("org") or {}
    if org.get("statusCode") != "LIQUIDATION_STAGE":
        continue
    dt = org.get("statusDate")
    if not dt:
        continue
    events.append({"inn": rec["inn"], "cls": "liquidation", "date": dt[:10],
                   "title": "ЕГРЮЛ: LIQUIDATION_STAGE", "guid": None})
    added += 1

json.dump(events, open(os.path.join(BASE, "events.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
print(f"added {added} liquidation events; total events {len(events)}")
