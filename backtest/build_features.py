"""Point-in-time features from ГИР БО: only statements published on or before as_of are used."""
import glob
import gzip
import json
import os
from datetime import date, datetime

import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW = os.path.join(BASE, "bfo_raw")

# as_of dates for the two folds
from panel_config import AS_OF, SUFFIX


def f(d, key):
    v = d.get(key)
    return float(v) if v is not None else None


def safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def parse_dt(s):
    if not s:
        return None
    s = s[:10]
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def features_from_detail(det, period_year, pub_date, as_of):
    bal = det.get("balance") or {}
    fin = det.get("financialResult") or {}
    cf = det.get("fundsMovement") or {}

    def b(code, pref="current"):
        return f(bal, f"{pref}{code}")

    def p(code, pref="current"):
        return f(fin, f"{pref}{code}")

    assets = b(1600)
    equity = b(1300)
    lt = b(1400) or 0.0
    st = b(1500) or 0.0
    debt_lt = b(1410) or 0.0
    debt_st = b(1510) or 0.0
    debt = debt_lt + debt_st
    cash = b(1250) or 0.0
    recv = b(1230)
    inv = b(1210)
    ca = b(1200)
    ap = b(1520)

    rev = p(2110)
    rev_prev = p(2110, "previous")
    gross = p(2100)
    opprofit = p(2200)
    interest = p(2330)
    net = p(2400)

    ocf = f(cf, "current4100")
    borrow_in = f(cf, "current4311")
    borrow_out = f(cf, "current4321")

    row = {
        "period": period_year,
        "pub_date": pub_date.isoformat() if pub_date else None,
        "pub_lag_days": (pub_date - date(period_year, 12, 31)).days if pub_date else None,
        "stale_days": (as_of - pub_date).days if pub_date else None,
        "assets": assets,
        "revenue": rev,
        "equity_ratio": safe_div(equity, assets),
        "neg_equity": 1 if (equity is not None and equity < 0) else 0,
        "debt_to_assets": safe_div(debt, assets),
        "liab_to_equity": safe_div(lt + st, equity) if (equity or 0) > 0 else None,
        "debt_to_ebit": safe_div(debt - cash, opprofit) if (opprofit or 0) > 0 else None,
        "icr": safe_div(opprofit, interest) if (interest or 0) > 0 else None,
        "no_interest_paid": 1 if not interest else 0,
        "current_ratio": safe_div(ca, st),
        "quick_ratio": safe_div((recv or 0) + cash, st),
        "cash_to_assets": safe_div(cash, assets),
        "cash_to_debt": safe_div(cash, debt) if debt else None,
        "recv_share": safe_div(recv, assets),
        "inv_share": safe_div(inv, assets),
        "ap_share": safe_div(ap, assets),
        "short_debt_share": safe_div(debt_st, debt) if debt else None,
        "roa": safe_div(net, assets),
        "op_margin": safe_div(opprofit, rev),
        "gross_margin": safe_div(gross, rev),
        "net_loss": 1 if (net is not None and net < 0) else 0,
        "rev_growth": safe_div(rev, rev_prev) if (rev_prev or 0) > 0 else None,
        "ocf_to_debt": safe_div(ocf, debt) if debt else None,
        "ocf_negative": 1 if (ocf is not None and ocf < 0) else 0,
        "refi_ratio": safe_div(borrow_in, borrow_out) if (borrow_out or 0) > 0 else None,
        "audit_required": det.get("requiredAudit"),
        "has_audit": 1 if (det.get("bfoFullness") or {}).get("hasAz") else 0,
        "corrections": det.get("correctionVersion"),
        "auditor_inn": ((det.get("auditReport") or {}).get("inn")),
    }
    return row


def pick_statement(rec, as_of):
    """Latest statement published on or before as_of, or None. Undated statements are skipped."""
    periods = {str(pp.get("period")): pp for pp in (rec.get("periods") or [])}
    best = None
    for year_s, det in (rec.get("details") or {}).items():
        year = int(year_s)
        pub = parse_dt((periods.get(year_s) or {}).get("actualBfoDate")) or parse_dt(
            det.get("datePresent")
        )
        if pub is None or pub > as_of:
            continue
        if best is None or year > best[0]:
            best = (year, det, pub)
    return best


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(RAW, "*.json.gz")))
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rec = json.load(fh)
        inn = rec["inn"]
        org = rec.get("org") or {}
        periods = {str(pp.get("period")): pp for pp in (rec.get("periods") or [])}

        for as_of in AS_OF:
            best = pick_statement(rec, as_of)
            if best is None:
                continue
            year, det, pub = best
            oi = det.get("bfoOrganizationInfo") or {}
            row = features_from_detail(det, year, pub, as_of)
            row.update(
                {
                    "inn": inn,
                    "as_of": as_of.isoformat(),
                    "name": rec.get("moex_name"),
                    "full_name": oi.get("fullName"),
                    "okved": org.get("okved2"),
                    "okved_name": ((oi.get("okved2") or {}) or {}).get("name")
                    if isinstance(oi.get("okved2"), dict)
                    else None,
                    "okopf_id": org.get("okopf"),
                    "region": org.get("region"),
                    "status": org.get("statusCode"),
                    "status_date": org.get("statusDate"),
                    "is_cb": int(bool((periods.get(str(year)) or {}).get("isCb"))),
                    "msp_category": (periods.get(str(year)) or {}).get("mspCategory"),
                    "n_filings": len(periods),
                }
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    print("rows:", len(df), "issuers:", df["inn"].nunique() if len(df) else 0)
    if len(df):
        print(df.groupby("as_of").size())
        df.to_csv(os.path.join(BASE, "features%s.csv" % SUFFIX), index=False, encoding="utf-8")
        print("saved features.csv")


if __name__ == "__main__":
    main()
