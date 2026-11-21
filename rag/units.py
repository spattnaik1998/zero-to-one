"""Units of the analysis tables' columns and summary values, read from their names.

The notebooks name every scaled quantity with a suffix: `gain_bn` (billions), `total_float_tn`
(trillions), `emp_2014_m` (millions), `mean_pay_2014_k` (thousands), `cagr_pct` (percent). The
suffix is the only record of the scale, so the tools report it (a model that sees 13.1 in
`op_income_bn` should write "$13.1 billion") and the answer verifier uses it (13.1 in a billions
column does not support "13.1 million").

A name without one of these suffixes has no known unit: the verifier then stays lenient about
scale, because a renamed or computed column (`SELECT gain_bn * 1e9 AS gain`) loses its suffix.
"""
from __future__ import annotations

import re

# suffix -> (unit label, multiplier from the stored value to the absolute amount)
SUFFIXES = {
    "tn": ("trillions", 1e12),
    "bn": ("billions", 1e9),
    "m": ("millions", 1e6),
    "k": ("thousands", 1e3),
    "pct": ("percent", None),
}
_SUFFIX = re.compile(r"_(tn|bn|m|k|pct)(?:_[a-z]{1,3})?$")      # emp_growth_pct_yr -> pct


def column_unit(name: str) -> dict | None:
    """{"unit": "billions", "scale": 1e9} for `op_income_bn`; {"unit": "percent", "scale": None}
    for `margin_pct` or `emp_growth_pct_yr`; None when the name carries no unit."""
    m = _SUFFIX.search(str(name).rsplit(".", 1)[-1].lower())
    if not m:
        return None
    unit, scale = SUFFIXES[m.group(1)]
    return {"unit": unit, "scale": scale}
