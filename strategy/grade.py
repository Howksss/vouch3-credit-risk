"""Instrument risk grade: rule-based decision strategy with versioned rule sets.

    rules_v1 — sector + three term flags
    rules_v2 — term penalties + issuer credit proxy (identity, track record)

Every grade carries model_version and a factor log, so a stored grade can always be
traced back to the rules and inputs that produced it. Pure functions over plain
dataclasses: no ORM, no database. The only input rules_v2 needs from storage is the
issuer track record, which is one SQL aggregate (issuer_stats.sql).
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ISSUER_STATS_SQL = Path(__file__).with_name("issuer_stats.sql").read_text(encoding="utf-8")

ISSUER_KEYWORDS_BANK = ("БАНК", "BANK")
ISSUER_KEYWORDS_DEVELOPER = ("СТРОЙ", "ДЕВЕЛОП", "НЕДВИЖ", "РИЭЛТ", "РИЕЛТ")
ISSUER_KEYWORDS_RETAIL = ("ТОРГ", "РИТЕЙЛ", "RETAIL", "МАРКЕТ", "MARKET")
ISSUER_KEYWORDS_INDUSTRIAL = ("ЗАВОД", "ПРОМЫШЛЕН", "МЕТАЛЛ", "НЕФТ", "ЭНЕРГ", "АГРО", "АГРА")

GRADE_LABELS = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-"]

SECTOR_SCORES: dict[str, float] = {
    "Bank": 0.0,
    "Industrial": 1.0,
    "Retailer": 1.5,
    "Developer": 2.5,
    "Other": 2.0,
}


@dataclass(frozen=True)
class Issuer:
    name: str = ""
    sector: str | None = None
    inn: str | None = None
    ogrn: str | None = None


@dataclass(frozen=True)
class Instrument:
    id: int | None = None
    issuer: Issuer | None = None
    coupon_rate: float | None = None
    face_value: float | None = None
    term_days: int | None = None


@dataclass(frozen=True)
class Grade:
    instrument_id: int | None
    model_version: str
    score: float
    grade_label: str
    factors: dict = field(default_factory=dict)


def classify_sector(issuer_name: str) -> str:
    upper = issuer_name.upper()
    if any(kw in upper for kw in ISSUER_KEYWORDS_BANK):
        return "Bank"
    if any(kw in upper for kw in ISSUER_KEYWORDS_DEVELOPER):
        return "Developer"
    if any(kw in upper for kw in ISSUER_KEYWORDS_RETAIL):
        return "Retailer"
    if any(kw in upper for kw in ISSUER_KEYWORDS_INDUSTRIAL):
        return "Industrial"
    return "Other"


def compute_grade(instrument: Instrument) -> Grade:
    """rules_v1 — terms + sector only. Kept for backward compatibility."""
    issuer_name = instrument.issuer.name if instrument.issuer else ""
    sector = (
        instrument.issuer.sector
        if instrument.issuer and instrument.issuer.sector
        else classify_sector(issuer_name)
    )

    score = SECTOR_SCORES.get(sector, 2.0)

    if instrument.coupon_rate is not None and instrument.coupon_rate > 25:
        score += 1.0
    if instrument.face_value is not None and instrument.face_value >= 1_000_000:
        score -= 1.0
    if instrument.term_days is not None and instrument.term_days < 90:
        score += 1.0

    idx = max(0, min(int(score), len(GRADE_LABELS) - 1))

    factors = {
        "sector": sector,
        "coupon_rate": float(instrument.coupon_rate) if instrument.coupon_rate else None,
        "face_value": float(instrument.face_value) if instrument.face_value else None,
        "term_days": instrument.term_days,
    }

    return Grade(
        instrument_id=instrument.id,
        model_version="rules_v1",
        score=float(idx),
        grade_label=GRADE_LABELS[idx],
        factors=factors,
    )


def _score_terms(instrument: Instrument) -> tuple[dict, dict]:
    """Term-based risk scoring: coupon rate, face value, duration."""
    penalties: dict[str, float] = {}
    details: dict = {}

    rate = float(instrument.coupon_rate) if instrument.coupon_rate is not None else None
    face = float(instrument.face_value) if instrument.face_value is not None else None
    term = instrument.term_days

    if rate is not None:
        details["coupon_rate"] = rate
        if rate > 30:
            penalties["coupon>30"] = 2.0
            details["coupon_flag"] = "very_high"
        elif rate > 25:
            penalties["coupon>25"] = 1.0
            details["coupon_flag"] = "high"

    if face is not None:
        details["face_value"] = face
        if face >= 1_000_000:
            penalties["face>=1m"] = -0.5
            details["face_flag"] = "institutional"

    if term is not None:
        details["term_days"] = term
        if term < 90:
            penalties["term<90d"] = 0.5
            details["term_flag"] = "ultra_short"
        elif term > 1095:
            penalties["term>1095d"] = -0.5
            details["term_flag"] = "long_duration"

    return penalties, details


def _score_issuer_credit(
    instrument: Instrument,
    issuer_stats: Mapping | None,
) -> tuple[dict, dict]:
    """Internal credit proxy — issuer track record from our own data."""
    penalties: dict[str, float] = {}
    details: dict = {}

    issuer = instrument.issuer
    if not issuer:
        details["identity"] = "unknown"
        penalties["issuer_unknown"] = 3.0
        return penalties, details

    issuer_name = issuer.name or ""
    sector = issuer.sector if issuer.sector else classify_sector(issuer_name)
    details["sector"] = sector
    penalties[f"sector={sector}"] = SECTOR_SCORES.get(sector, 2.0)

    if issuer.ogrn:
        penalties["ogrn_verified"] = -0.5
        details["identity"] = "ogrn_verified"
    elif issuer.inn:
        penalties["inn_verified"] = -0.25
        details["identity"] = "inn_verified"
    else:
        penalties["name_only"] = 0.5
        details["identity"] = "name_only"

    if issuer_stats:
        total = issuer_stats.get("total_issues", 0)
        active = issuer_stats.get("active_issues", 0)
        redeemed = issuer_stats.get("redeemed_issues", 0)
        defaulted = issuer_stats.get("defaulted_issues", 0)
        cancelled = issuer_stats.get("cancelled_issues", 0)

        details["total_issues"] = total
        details["active_issues"] = active
        details["redeemed_issues"] = redeemed

        if defaulted > 0:
            penalties["defaulted_issues"] = 3.0
            details["default_flag"] = f"{defaulted}_defaults"

        if cancelled > 0:
            penalties["cancelled_issues"] = 0.5 * min(cancelled, 3)
            details["cancel_flag"] = f"{cancelled}_cancelled"

        if redeemed >= 3:
            penalties["redeemed>=3"] = -1.0
            details["track_record"] = "proven"
        elif redeemed >= 1:
            penalties["redeemed>=1"] = -0.5
            details["track_record"] = "some_history"
        elif total == 1:
            penalties["first_issue"] = 0.5
            details["track_record"] = "first_issue"

        avg_rate = issuer_stats.get("avg_coupon")
        if avg_rate is not None:
            details["issuer_avg_coupon"] = round(avg_rate, 2)

    return penalties, details


def compute_grade_v2(
    instrument: Instrument,
    issuer_stats: Mapping | None = None,
) -> Grade:
    """rules_v2 — terms + internal credit proxy (issuer track record from own data)."""
    term_penalties, term_factors = _score_terms(instrument)
    credit_penalties, credit_factors = _score_issuer_credit(instrument, issuer_stats)

    penalties = {**term_penalties, **credit_penalties}
    raw_score = sum(penalties.values())
    idx = max(0, min(int(round(raw_score)), len(GRADE_LABELS) - 1))

    factors = {
        "terms": term_factors,
        "credit": credit_factors,
        "penalties": penalties,
        "raw_score": round(raw_score, 2),
    }

    return Grade(
        instrument_id=instrument.id,
        model_version="rules_v2",
        score=float(idx),
        grade_label=GRADE_LABELS[idx],
        factors=factors,
    )


def issuer_stats(row: Mapping) -> dict:
    """One row of ISSUER_STATS_SQL -> the issuer_stats mapping rules_v2 takes."""
    stats = dict(row)
    if stats.get("avg_coupon") is not None:
        stats["avg_coupon"] = float(stats["avg_coupon"])
    return stats
