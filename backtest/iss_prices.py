"""Close price and yield per issuer at each as_of from MOEX ISS, one bulk call per trading day."""
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CTX = ssl.create_default_context()
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"

from panel_config import AS_OF, SUFFIX  # noqa: E402

# no data on non-trading days, step back up to 6 days
DATES = {d.isoformat(): d for d in AS_OF}


def get(url, tries=5):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, context=CTX, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("giving up on " + url)


sec = json.load(open(os.path.join(BASE, "moex_securities.json"), encoding="utf-8"))
sec2inn = {s["secid"]: (s.get("emitent_inn") or "").strip() for s in sec}

out = []
from datetime import timedelta  # noqa: E402

for as_of, as_of_date in DATES.items():
    rows = []
    trade_date = None
    for back in range(7):
        cand = (as_of_date - timedelta(days=back)).isoformat()
        start = 0
        rows = []
        while True:
            url = ("https://iss.moex.com/iss/history/engines/stock/markets/bonds/securities.json"
                   f"?date={cand}&iss.meta=off&limit=100&start={start}")
            d = get(url)["history"]
            if not d["data"]:
                break
            cols = d["columns"]
            rows.extend(dict(zip(cols, r)) for r in d["data"])
            start += 100
            time.sleep(0.05)
        if rows:
            trade_date = cand
            break
    print(f"{as_of}: {len(rows)} quotes on {trade_date}", flush=True)

    per_issuer = defaultdict(list)
    for r in rows:
        inn = sec2inn.get(r.get("SECID"))
        if not inn:
            continue
        price = r.get("CLOSE") or r.get("LEGALCLOSEPRICE") or r.get("MARKETPRICE2")
        yld = r.get("YIELDCLOSE") or r.get("YIELDATWAP")
        if price is None and yld is None:
            continue
        per_issuer[inn].append((price, yld, r.get("DURATION"), r.get("NUMTRADES") or 0))

    for inn, items in per_issuer.items():
        prices = [p for p, _, _, _ in items if p]
        ylds = [y for _, y, _, _ in items if y is not None]
        durs = [d for _, _, d, _ in items if d]
        out.append({
            "inn": inn,
            "as_of": as_of,
            "n_quoted_bonds": len(items),
            "min_price": min(prices) if prices else None,
            "med_price": sorted(prices)[len(prices) // 2] if prices else None,
            "max_yield": max(ylds) if ylds else None,
            "med_yield": sorted(ylds)[len(ylds) // 2] if ylds else None,
            "med_duration": sorted(durs)[len(durs) // 2] if durs else None,
            "trades": sum(n for _, _, _, n in items),
        })

import pandas as pd  # noqa: E402

df = pd.DataFrame(out)
print("\nrows:", len(df), "issuers:", df["inn"].nunique())
print(df.groupby("as_of").size())
df.to_csv(os.path.join(BASE, "market%s.csv" % SUFFIX), index=False, encoding="utf-8")
print("saved market.csv")
