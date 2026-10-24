"""Shared as_of grid. `--monthly` switches the builders to the hazard panel."""
import sys
from datetime import date


def month_ends(start_year, start_month, end_year, end_month):
    out = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        if m == 12:
            nxt = date(y + 1, 1, 1)
        else:
            nxt = date(y, m + 1, 1)
        out.append(date.fromordinal(nxt.toordinal() - 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


SEMI = [date(2023, 6, 30), date(2023, 12, 31), date(2024, 6, 30),
        date(2024, 12, 31), date(2025, 6, 30), date(2025, 12, 31)]

MONTHLY = month_ends(2023, 1, 2025, 12)

IS_MONTHLY = "--monthly" in sys.argv
AS_OF = MONTHLY if IS_MONTHLY else SEMI
SUFFIX = "_monthly" if IS_MONTHLY else ""
