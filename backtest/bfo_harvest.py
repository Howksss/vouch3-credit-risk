"""Harvest ГИР БО statements for every MOEX bond issuer INN. Resumable via bfo_state.json."""
import gzip
import json
import os
import random
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW = os.path.join(BASE, "bfo_raw")
os.makedirs(RAW, exist_ok=True)

CTX = ssl.create_default_context()
H = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://bo.nalog.gov.ru/",
}
MIN_PERIOD = 2020
lock = threading.Lock()
state_path = os.path.join(BASE, "bfo_state.json")
state = json.load(open(state_path, encoding="utf-8")) if os.path.exists(state_path) else {}


def get(url, tries=4):
    for attempt in range(tries):
        req = urllib.request.Request(url, headers=H)
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504):
                time.sleep(3 * (attempt + 1) + random.random() * 2)
                continue
            return None
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return None


def work(item):
    inn = item["inn"]
    if state.get(inn, {}).get("ok"):
        return
    if not (len(inn) == 10 and inn.isdigit()):
        with lock:
            state[inn] = {"ok": False, "why": "bad_inn"}
        return

    s = get(f"https://bo.nalog.gov.ru/advanced-search/organizations/search?query={inn}&page=0")
    time.sleep(0.25 + random.random() * 0.25)
    if not s or not s.get("content"):
        with lock:
            state[inn] = {"ok": False, "why": "not_found"}
        return
    org = s["content"][0]
    oid = org["id"]

    periods = get(f"https://bo.nalog.gov.ru/nbo/organizations/{oid}/bfo/") or []
    time.sleep(0.25 + random.random() * 0.25)

    details = {}
    for p in periods:
        try:
            year = int(p.get("period") or 0)
        except (TypeError, ValueError):
            continue
        if year < MIN_PERIOD:
            continue
        d = get(f"https://bo.nalog.gov.ru/nbo/bfo/{p['id']}/details")
        time.sleep(0.25 + random.random() * 0.25)
        if d:
            details[str(year)] = d[0] if isinstance(d, list) and d else d

    payload = {"inn": inn, "moex_name": item.get("name"), "org": org, "periods": periods,
               "details": details}
    with gzip.open(os.path.join(RAW, f"{inn}.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    with lock:
        state[inn] = {"ok": True, "n_periods": len(periods), "n_details": len(details)}
        done = len(state)
        if done % 25 == 0:
            json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"{done} done, last {inn} periods={len(periods)}", flush=True)


issuers = json.load(open(os.path.join(BASE, "moex_issuers.json"), encoding="utf-8"))
todo = [e for e in issuers if not state.get(e["inn"], {}).get("ok")]
print(f"universe={len(issuers)} todo={len(todo)}", flush=True)

with ThreadPoolExecutor(max_workers=6) as ex:
    list(ex.map(work, todo))

json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False)
ok = sum(1 for v in state.values() if v.get("ok"))
print(f"DONE ok={ok}/{len(state)}", flush=True)
