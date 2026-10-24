"""Harvest the full 1prime disclosure history for every MOEX bond issuer INN.

Output: prime_messages.jsonl  (one row per disclosure message)
        prime_state.json      (per-INN fetch status, resumable)
"""
import gzip
import json
import os
import random
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW = os.path.join(BASE, "prime_raw")
os.makedirs(RAW, exist_ok=True)

# disclosure.1prime.ru chains to the HARICA ECC root, which is missing from some
# system trust stores (Windows among them); the pages are public and read-only
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

ROW = re.compile(
    r"<tr><td valign='top'>(\d+)</td>"
    r"<td width='50%'[^>]*>(.*?)</td>"
    r"<td align='center'>([\d.]*)</td>"
    r"<td align='center'>([\d.]*)</td>"
    r"<td align='center'>(.*?)</td>",
    re.S,
)
GUID = re.compile(r"guid=(\{[^}]+\})", re.I)
TAG = re.compile(r"<[^>]+>")

lock = threading.Lock()
out_f = open(os.path.join(BASE, "prime_messages.jsonl"), "a", encoding="utf-8")
state_path = os.path.join(BASE, "prime_state.json")
state = json.load(open(state_path, encoding="utf-8")) if os.path.exists(state_path) else {}


def clean(s):
    import html as h

    return h.unescape(TAG.sub("", s)).strip()


def fetch(inn, tries=4):
    url = f"https://disclosure.1prime.ru/portal/default.aspx?emId={inn}"
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=90) as r:
                return r.read().decode("cp1251", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (503, 429, 502):
                time.sleep(3 * (attempt + 1) + random.random() * 2)
                continue
            return None
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return None


def work(item):
    inn, name = item["inn"], item["name"]
    if state.get(inn, {}).get("ok"):
        return
    html_text = fetch(inn)
    time.sleep(0.4 + random.random() * 0.4)
    if html_text is None:
        with lock:
            state[inn] = {"ok": False, "n": 0}
        return
    with gzip.open(os.path.join(RAW, f"{inn}.html.gz"), "wt", encoding="utf-8") as f:
        f.write(html_text)
    rows = ROW.findall(html_text)
    recs = []
    for _num, title, d_feed, d_site, link in rows:
        g = GUID.search(link)
        recs.append(
            {
                "inn": inn,
                "issuer": name,
                "title": clean(title),
                "date_feed": d_feed,
                "date_site": d_site,
                "guid": g.group(1) if g else None,
            }
        )
    with lock:
        for r in recs:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_f.flush()
        state[inn] = {"ok": True, "n": len(recs)}
        done = len(state)
        if done % 25 == 0:
            json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"{done} issuers done, last {inn} n={len(recs)}", flush=True)


issuers = json.load(open(os.path.join(BASE, "moex_issuers.json"), encoding="utf-8"))
todo = [e for e in issuers if not state.get(e["inn"], {}).get("ok")]
# issuers that already have ГИР БО statements are the ones the model can use —
# fetch their disclosures first so the dataset is usable before the tail finishes
have_bfo = {f[:-8] for f in os.listdir(os.path.join(BASE, "bfo_raw"))}
todo.sort(key=lambda e: 0 if e["inn"] in have_bfo else 1)
print(f"universe={len(issuers)} todo={len(todo)} "
      f"with_bfo_first={sum(1 for e in todo if e['inn'] in have_bfo)}", flush=True)

with ThreadPoolExecutor(max_workers=5) as ex:
    list(ex.map(work, todo))

json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False)
ok = sum(1 for v in state.values() if v["ok"])
print(f"DONE ok={ok}/{len(state)} messages={sum(v['n'] for v in state.values())}", flush=True)
