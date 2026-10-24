"""The anti-leak rules of the backtest, checked on toy inputs."""
from datetime import date
from itertools import product

import numpy as np
import pandas as pd

import model as M
from build_features import pick_statement


def _rec(**statements):
    """ГИР БО record: {year: (actualBfoDate, datePresent)}."""
    return {
        "periods": [{"period": y[1:], "actualBfoDate": a} for y, (a, _) in statements.items()],
        "details": {y[1:]: {"datePresent": p} for y, (_, p) in statements.items()},
    }


class TestPointInTimeStatements:
    rec = _rec(y2023=("2024-04-20", None), y2024=("2025-04-15", None))

    def test_statement_published_after_as_of_is_not_used(self) -> None:
        # FY2024 is filed in April 2025: invisible at 31.03.2025, visible at 30.06.2025
        assert pick_statement(self.rec, date(2025, 3, 31))[0] == 2023
        assert pick_statement(self.rec, date(2025, 6, 30))[0] == 2024

    def test_period_end_is_not_the_availability_date(self) -> None:
        assert pick_statement(self.rec, date(2024, 12, 31))[0] == 2023
        assert pick_statement(self.rec, date(2024, 3, 31)) is None

    def test_falls_back_to_date_present(self) -> None:
        rec = _rec(y2024=(None, "15.05.2025"))
        assert pick_statement(rec, date(2025, 5, 14)) is None
        assert pick_statement(rec, date(2025, 5, 15))[2] == date(2025, 5, 15)

    def test_statement_without_any_date_is_skipped(self) -> None:
        assert pick_statement(_rec(y2024=(None, None)), date(2030, 1, 1)) is None


class TestLabelWindow:
    t = date(2025, 12, 31)

    def test_event_on_snapshot_day_is_not_a_future_event(self) -> None:
        assert M.has_event([self.t], self.t, 250) == 0

    def test_window_is_closed_at_the_horizon(self) -> None:
        assert M.has_event([date(2026, 9, 7)], self.t, 250) == 1
        assert M.has_event([date(2026, 9, 8)], self.t, 250) == 0


class TestSplit:
    def test_train_label_windows_close_before_test_snapshot(self) -> None:
        snaps = ["2023-06-30", "2023-12-31", "2024-06-30", "2024-12-31",
                 "2025-06-30", "2025-12-31"]
        u = pd.DataFrame({"as_of": snaps, "y": 0})
        train, test = M.split(u, 250)
        assert list(test["as_of"]) == ["2025-12-31"]
        # 30.06.2025 + 250 days runs into 2026: its labels overlap the test window
        assert list(train["as_of"]) == snaps[:4]


class TestUniverse:
    def test_drops_unobservable_financial_and_already_defaulted(self) -> None:
        base = {"observable": 1, "is_spv": 0, "is_fin": 0, "is_cb": 0, "default_prior": 0}
        rows = [base] + [{**base, k: 1 - base[k]} for k in base]
        assert len(M.universe(pd.DataFrame(rows))) == 1


class TestAuc:
    def test_matches_pairwise_definition_with_ties(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 60)
        s = rng.integers(0, 6, 60).astype(float)
        pos, neg = s[y == 1], s[y == 0]
        pairs = [1.0 if a > b else 0.5 if a == b else 0.0 for a, b in product(pos, neg)]
        assert np.isclose(M.auc(y, s), np.mean(pairs))
