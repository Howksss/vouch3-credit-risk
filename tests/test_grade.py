import sqlite3

import pytest

from strategy.grade import (
    GRADE_LABELS,
    ISSUER_STATS_SQL,
    Instrument,
    Issuer,
    classify_sector,
    compute_grade,
    compute_grade_v2,
    issuer_stats,
)


class TestClassifySector:
    def test_bank(self) -> None:
        assert classify_sector("АО ТЕСТ-БАНК") == "Bank"
        assert classify_sector("ПАО ПРИМЕРБАНК") == "Bank"

    def test_developer(self) -> None:
        assert classify_sector("ООО СТРОЙМОНТАЖ") == "Developer"
        assert classify_sector("АО ДЕВЕЛОПМЕНТ") == "Developer"

    def test_retailer(self) -> None:
        assert classify_sector("ООО ТОРГОВЫЙ ДОМ") == "Retailer"
        assert classify_sector("АО РИТЕЙЛ ГРУПП") == "Retailer"

    def test_industrial(self) -> None:
        assert classify_sector("ОАО ЗАВОД ТЕСТМАШ") == "Industrial"
        assert classify_sector("ООО НЕФТЕГАЗ") == "Industrial"

    def test_other(self) -> None:
        assert classify_sector("ООО ПРИМЕР") == "Other"
        assert classify_sector("ЗАО ПРИМЕР-ПЛЮС") == "Other"


class TestGradesList:
    def test_grades_count(self) -> None:
        assert len(GRADE_LABELS) == 12

    def test_grades_order(self) -> None:
        assert GRADE_LABELS[0] == "A+"
        assert GRADE_LABELS[-1] == "D-"


class TestRulesV1:
    def test_institutional_bank_bond_is_top_grade(self) -> None:
        g = compute_grade(Instrument(id=1, issuer=Issuer(name="АО ТЕСТ-БАНК"),
                                     coupon_rate=12.0, face_value=1_000_000, term_days=365))
        assert (g.model_version, g.grade_label) == ("rules_v1", "A+")

    def test_short_high_coupon_developer(self) -> None:
        # Developer 2.5 + coupon>25 1.0 + term<90 1.0 = 4.5, floored to 4
        g = compute_grade(Instrument(issuer=Issuer(name="ООО СТРОЙМОНТАЖ"),
                                     coupon_rate=28.0, term_days=60))
        assert g.grade_label == "B"
        assert g.factors == {"sector": "Developer", "coupon_rate": 28.0,
                             "face_value": None, "term_days": 60}

    def test_explicit_sector_overrides_name(self) -> None:
        g = compute_grade(Instrument(issuer=Issuer(name="АО ТЕСТ-БАНК", sector="Developer")))
        assert g.factors["sector"] == "Developer"


class TestRulesV2:
    def test_factor_log_adds_up_to_score(self) -> None:
        g = compute_grade_v2(
            Instrument(id=7, issuer=Issuer(name="ООО ПРИМЕР", inn="7700000000"),
                       coupon_rate=32.0, term_days=365),
            {"total_issues": 1, "active_issues": 1, "redeemed_issues": 0},
        )
        assert g.factors["penalties"] == {
            "coupon>30": 2.0,
            "sector=Other": 2.0,
            "inn_verified": -0.25,
            "first_issue": 0.5,
        }
        assert sum(g.factors["penalties"].values()) == g.factors["raw_score"] == 4.25
        assert (g.instrument_id, g.model_version, g.grade_label) == (7, "rules_v2", "B")

    def test_unknown_issuer_skips_credit_block(self) -> None:
        g = compute_grade_v2(Instrument(coupon_rate=10.0),
                             {"total_issues": 5, "redeemed_issues": 5})
        assert g.factors["credit"] == {"identity": "unknown"}
        assert g.factors["penalties"] == {"issuer_unknown": 3.0}
        assert g.grade_label == "B+"

    def test_default_in_history_costs_three_notches(self) -> None:
        inst = Instrument(issuer=Issuer(name="ООО ПРИМЕР", ogrn="1027700000000"),
                          coupon_rate=20.0, term_days=720)
        clean = compute_grade_v2(inst, {"total_issues": 2, "redeemed_issues": 1})
        hit = compute_grade_v2(inst, {"total_issues": 2, "redeemed_issues": 1,
                                      "defaulted_issues": 1})
        assert hit.score - clean.score == 3
        assert hit.factors["credit"]["default_flag"] == "1_defaults"

    def test_cancelled_penalty_is_capped(self) -> None:
        g = compute_grade_v2(Instrument(issuer=Issuer(name="ООО ПРИМЕР")),
                             {"total_issues": 9, "cancelled_issues": 7})
        assert g.factors["penalties"]["cancelled_issues"] == 1.5

    def test_score_floors_at_best_grade(self) -> None:
        g = compute_grade_v2(
            Instrument(issuer=Issuer(name="АО ТЕСТ-БАНК", ogrn="1027700000000"),
                       face_value=1_000_000, term_days=1800),
            {"total_issues": 6, "redeemed_issues": 5},
        )
        assert g.factors["raw_score"] == -2.5
        assert g.grade_label == "A+"

    def test_versions_are_not_interchangeable(self) -> None:
        inst = Instrument(issuer=Issuer(name="ООО ПРИМЕР", inn="7700000000"),
                          coupon_rate=32.0, term_days=365)
        v1, v2 = compute_grade(inst), compute_grade_v2(inst, {"total_issues": 1})
        assert (v1.model_version, v1.grade_label) == ("rules_v1", "B+")
        assert (v2.model_version, v2.grade_label) == ("rules_v2", "B")


class TestIssuerStatsSql:
    @pytest.fixture
    def db(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE instrument (id INTEGER PRIMARY KEY, issuer_id INTEGER,"
                     " status TEXT, coupon_rate REAL)")
        conn.executemany(
            "INSERT INTO instrument (issuer_id, status, coupon_rate) VALUES (?, ?, ?)",
            [(1, "redeemed", 14.0), (1, "redeemed", 16.0), (1, "redeemed", None),
             (1, "active", 18.0), (1, "cancelled", None), (2, "defaulted", 30.0)],
        )
        yield conn
        conn.close()

    def _stats(self, conn, issuer_id: int) -> dict:
        cur = conn.execute(ISSUER_STATS_SQL, {"issuer_id": issuer_id})
        return issuer_stats(dict(zip([c[0] for c in cur.description], cur.fetchone())))

    def test_aggregate(self, db) -> None:
        assert self._stats(db, 1) == {
            "total_issues": 5, "active_issues": 1, "redeemed_issues": 3,
            "defaulted_issues": 0, "cancelled_issues": 1, "avg_coupon": 16.0,
        }

    def test_feeds_rules_v2(self, db) -> None:
        inst = Instrument(issuer=Issuer(name="ООО ПРИМЕР", inn="7700000000"), coupon_rate=15.0)
        good = compute_grade_v2(inst, self._stats(db, 1))
        bad = compute_grade_v2(inst, self._stats(db, 2))
        assert good.factors["credit"]["track_record"] == "proven"
        assert bad.factors["penalties"]["defaulted_issues"] == 3.0
        assert GRADE_LABELS.index(bad.grade_label) > GRADE_LABELS.index(good.grade_label)
