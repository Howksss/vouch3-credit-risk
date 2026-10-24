"""Pull the full MOEX bond universe from ISS with issuer INNs."""
import json
import os
import ssl
import time
import urllib.request

CTX = ssl.create_default_context()
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, context=CTX, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


rows = []
start = 0
while True:
    url = (
        "https://iss.moex.com/iss/securities.json?engine=stock&market=bonds"
        f"&iss.meta=off&limit=100&start={start}"
    )
    d = get(url)["securities"]
    cols = d["columns"]
    data = d["data"]
    if not data:
        break
    rows.extend(dict(zip(cols, r)) for r in data)
    start += 100
    if start % 1000 == 0:
        print("fetched", start, flush=True)
    time.sleep(0.15)

print("total securities:", len(rows))

issuers = {}
for r in rows:
    inn = (r.get("emitent_inn") or "").strip()
    if not inn:
        continue
    e = issuers.setdefault(
        inn,
        {
            "inn": inn,
            "emitent_id": r.get("emitent_id"),
            "name": r.get("emitent_title"),
            "n_sec": 0,
            "n_traded": 0,
            "types": set(),
        },
    )
    e["n_sec"] += 1
    e["n_traded"] += 1 if r.get("is_traded") else 0
    e["types"].add(r.get("type"))

for e in issuers.values():
    e["types"] = sorted(t for t in e["types"] if t)

print("distinct issuer INNs:", len(issuers))
print("with >=1 traded sec:", sum(1 for e in issuers.values() if e["n_traded"]))

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(base, exist_ok=True)
with open(os.path.join(base, "moex_securities.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False)
with open(os.path.join(base, "moex_issuers.json"), "w", encoding="utf-8") as f:
    json.dump(list(issuers.values()), f, ensure_ascii=False, indent=1)
print("saved")
