"""Shared helpers for the Zero to One hypothesis notebooks.

Keys and contact details come from the nearest .env up the folder tree:
  FRED_API_KEY, CENSUS_API_KEY, SEC_USER_AGENT  (e.g. "Your Name you@example.com")
Downloads are cached in data/raw/. Delete a file there to refresh it.
Set THIEL_SYNTHETIC=1 to run on generated data (testing only).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
FIG = OUT / "figures"
TAB = OUT / "tables"
for _d in (RAW, FIG, TAB):
    _d.mkdir(parents=True, exist_ok=True)

SYNTHETIC = os.environ.get("THIEL_SYNTHETIC") == "1"
SCALE_FIXES = None
BOOK_DATE = pd.Timestamp("2014-09-16")       # US publication date of Zero to One

# Validated categorical palette (light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GREY, INK, MUTED = "#9a9a96", "#2b2b29", "#6b6b66"


# ------------------------------------------------------------------ .env
def load_env() -> dict:
    env = {}
    for d in [ROOT, *ROOT.parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*?)\s*[:=]\s*(.*)$", line)
                if m:
                    env[m.group(1).strip()] = m.group(2).strip().strip('"').strip("'")
            break
    return env


def _need(name: str) -> str:
    v = load_env().get(name) or os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is missing. Add a line '{name}=...' to Recent/.env")
    return v


# ------------------------------------------------------------------ SEC XBRL frames
SEC_FRAMES = "https://data.sec.gov/api/xbrl/frames/{tax}/{tag}/{unit}/{period}.json"


def sec_frame(tag: str, period: str, tax: str = "us-gaap", unit: str = "USD") -> pd.DataFrame:
    """One XBRL frame: every filer's value of `tag` for `period` (e.g. 'CY2014Q2I' or 'CY2014').
    Returns columns cik, entityName, end, val, accn. Empty frame if SEC has none."""
    cache = RAW / "sec" / f"{tax}_{tag}_{unit}_{period}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC:
        return _synthetic_frame(tag, period)
    if not cache.exists():
        import requests
        ua = _need("SEC_USER_AGENT")
        url = SEC_FRAMES.format(tax=tax, tag=tag, unit=unit, period=period)
        r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=60)
        time.sleep(0.15)                       # SEC asks for at most 10 requests per second
        if r.status_code == 404:
            cache.write_text('{"data": []}')
        else:
            r.raise_for_status()
            cache.write_bytes(r.content)
    data = json.loads(cache.read_text())["data"]
    df = pd.DataFrame(data, columns=["accn", "cik", "entityName", "loc", "end", "val"])
    df["end"] = pd.to_datetime(df["end"])
    return df


# Exchange-traded products that file 10-Ks but are not operating companies
_ETP = re.compile(r"\b(?:ETF|ETFs|FUND|FUNDS)\b|"
                  r"(?:GOLD|SILVER|BITCOIN|ETHER|ETHEREUM|PLATINUM|PALLADIUM|COMMODITY|CURRENCY|OIL|VIX)"
                  r".*\bTRUST\b|\b(?:ISHARES|SPDR|GRAYSCALE|PROSHARES|INVESCO DB|TEUCRIUM|WISDOMTREE)\b",
                  re.I)


def public_float_panel(years: range) -> pd.DataFrame:
    """dei:EntityPublicFloat for every SEC filer, one row per filer-year.

    Public float is the market value of shares held by non-affiliates, measured on the last
    business day of the filer's second fiscal quarter. Firms with different fiscal years report
    in different calendar quarters, so all four quarterly frames are combined and the latest
    value per filer within each calendar year is kept."""
    parts = []
    for y in years:
        for q in range(1, 5):
            f = sec_frame("EntityPublicFloat", f"CY{y}Q{q}I", tax="dei")
            if len(f):
                f["year"] = y
                parts.append(f)
    df = pd.concat(parts, ignore_index=True)
    df = df[(df.val > 0) & (df.val < 6e12)]                  # drop zeros and impossible values
    df = df[~df.entityName.fillna("").str.contains(_ETP.pattern, flags=re.I, regex=True)]
    df = df.sort_values("end").groupby(["cik", "year"], as_index=False).last()
    df, log = fix_scale_errors(df)
    if os.environ.get("THIEL_SKIP_ASSETS") == "1":           # diagnostics only; never for final results
        log2 = pd.DataFrame()
    else:
        df, log2 = check_against_assets(df, years)
    global SCALE_FIXES
    SCALE_FIXES = pd.concat([log, log2], ignore_index=True)   # inspect via tc.SCALE_FIXES
    df = link_successors(df)
    return df[["cik", "entityName", "year", "end", "val"]]


def total_assets(years) -> pd.Series:
    """us-gaap:Assets at fiscal year-end, indexed by (cik, year). Frames CY{y}Q4I hold the
    balance-sheet value closest to each calendar year-end."""
    parts = []
    for y in years:
        f = sec_frame("Assets", f"CY{y}Q4I")
        if len(f):
            parts.append(f.assign(year=y)[["cik", "year", "val"]])
    a = pd.concat(parts).groupby(["cik", "year"]).val.last()
    return a[a > 0]


def check_against_assets(df: pd.DataFrame, years, max_ratio: float = 150.0, min_float: float = 1e9):
    """Second screen for scale errors that are persistent within a firm (so its own history
    cannot reveal them), e.g. Biglari Holdings reporting ~$500bn of float in every year.

    For floats above `min_float`, compare with total assets in the same or previous year (the
    larger of the two). A float above `max_ratio` x assets is rescaled /1000 if that brings the
    ratio below `max_ratio`, otherwise dropped. Floats above $100bn with no assets on file are
    dropped (every genuine firm that size files a balance sheet)."""
    A = total_assets(range(min(years) - 1, max(years) + 1))
    a0 = A.reindex(pd.MultiIndex.from_arrays([df.cik, df.year])).values
    a1 = A.reindex(pd.MultiIndex.from_arrays([df.cik, df.year - 1])).values
    assets = np.fmax(a0, a1)
    ratio = df.val.values / assets
    big = df.val.values > min_float
    bad = big & (ratio > max_ratio)
    fixable = bad & (ratio / 1000 < max_ratio)
    orphan = (df.val.values > 1e11) & np.isnan(assets)
    action = np.where(fixable, "rescaled /1000 (assets check)",
                      np.where(bad | orphan, "dropped (assets check)", "ok"))
    df = df.copy()
    df.loc[fixable, "val"] = df.loc[fixable, "val"] / 1000
    log = df.assign(action=action, deviation=np.round(np.log10(ratio), 2))[action != "ok"][
        ["cik", "entityName", "year", "val", "deviation", "action"]]
    return df[action == "ok"].pipe(lambda d: pd.concat([d, df[fixable]])), log


# Old CIK -> successor CIK, for reorganisations and mergers that issued a new SEC registrant.
# Each link was checked in the float panel: the old CIK stops reporting when the new one starts.
SUCCESSOR = {
    1288776: 1652044,   # Google Inc. -> Alphabet Inc. (2015 holding-company reorganisation)
    1441634: 1649338,   # Avago Technologies -> Broadcom Ltd (2016)
    1649338: 1730168,   # Broadcom Ltd -> Broadcom Inc. (2018 redomicile)
    64670: 1613103,     # Medtronic Inc. -> Medtronic plc (2015, Covidien deal)
    884905: 1707925,    # Praxair -> Linde plc (2018 merger with Linde AG)
    1001039: 1744489,   # Walt Disney Co -> new Walt Disney Co (2019, 21st Century Fox deal)
    701221: 1739940,    # Cigna Corp -> Cigna Group (2018, Express Scripts deal)
    29915: 1666700,     # Dow Chemical -> DowDuPont / DuPont de Nemours (2017 merger)
    30554: 1666700,     # E. I. du Pont -> DowDuPont / DuPont de Nemours (2017 merger)
    53669: 833444,      # Johnson Controls Inc. -> Johnson Controls International plc (2016, Tyco)
    1545158: 1637459,   # Kraft Foods Group -> Kraft Heinz (2015)
}


def _final(cik):
    seen = set()
    while cik in SUCCESSOR and cik not in seen:
        seen.add(cik); cik = SUCCESSOR[cik]
    return cik


def link_successors(df: pd.DataFrame) -> pd.DataFrame:
    """Re-key predecessor filers to their final successor. In a year where the successor itself
    reports, its value is used (it already includes the predecessor); otherwise predecessors'
    values are summed (two merging firms count as one combined position)."""
    df = df.copy()
    df["orig_cik"] = df["cik"]
    df["cik"] = df["cik"].map(_final)
    own = df[df.cik == df.orig_cik]
    pred = df[df.cik != df.orig_cik]
    pred = pred[~pred.set_index(["cik", "year"]).index.isin(own.set_index(["cik", "year"]).index)]
    pred = pred.groupby(["cik", "year"], as_index=False).agg(
        entityName=("entityName", "last"), end=("end", "last"), val=("val", "sum"))
    return pd.concat([own.drop(columns="orig_cik"), pred], ignore_index=True)


def value_near(panel: pd.DataFrame, year: int, window=(0, -1, 1)) -> pd.Series:
    """Each filer's value in `year`, or else in the nearest year listed in `window`
    (offsets tried in order). Covers firms whose filing landed outside one year's frames."""
    piv = panel.pivot_table(index="cik", columns="year", values="val", aggfunc="last")
    out = pd.Series(np.nan, index=piv.index)
    for off in window:
        if year + off in piv.columns:
            out = out.fillna(piv[year + off])
    return out.dropna()


def fix_scale_errors(df: pd.DataFrame, thr: float = 2.5, tol: float = 1.0):
    """Repair filings tagged at the wrong scale (typically 1,000x too large or too small).

    For each firm, the reference level is the median log10 value over its years. A value more
    than `thr` orders of magnitude (about 300x) away from that reference is divided or multiplied
    by 1,000; the fix is kept if the result lies within `tol` orders of magnitude of the
    reference, otherwise the observation is dropped. Medians are recomputed after a first pass
    so that firms with several bad years are judged against their good years."""
    df = df.copy()
    lv = np.log10(df.val)
    ref = lv.groupby(df.cik).transform("median")
    # second pass: reference from values that are not obviously off
    good = (lv - ref).abs() < thr
    ref2 = lv.where(good).groupby(df.cik).transform("median").fillna(ref)
    dev = lv - ref2
    action = pd.Series("ok", index=df.index)
    for sign, factor in ((1, 1e-3), (-1, 1e3)):
        cand = sign * dev > thr
        fixed = np.log10(df.val * factor)
        okfix = cand & ((fixed - ref2).abs() < tol)
        df.loc[okfix, "val"] = df.loc[okfix, "val"] * factor
        action[okfix] = "rescaled x1000" if factor > 1 else "rescaled /1000"
        action[cand & ~okfix] = "dropped"
    log = df.assign(action=action, deviation=dev.round(2))[action != "ok"][
        ["cik", "entityName", "year", "val", "deviation", "action"]]
    return df[action != "dropped"], log


# ------------------------------------------------------------------ Y Combinator
YC_URL = "https://raw.githubusercontent.com/yc-oss/api/main/companies/all.json"
_SEASON = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}


def yc_companies() -> pd.DataFrame:
    cache = RAW / "yc_all.json"
    if not cache.exists():
        import requests
        r = requests.get(YC_URL, timeout=120)
        r.raise_for_status()
        cache.write_bytes(r.content)
    df = pd.DataFrame(json.loads(cache.read_text(encoding="utf-8")))
    m = df["batch"].fillna("").str.extract(r"(Winter|Spring|Summer|Fall)\s+(\d{4})")
    df["batch_year"] = pd.to_numeric(m[1], errors="coerce")
    df["batch_order"] = df["batch_year"] * 4 + m[0].map(_SEASON)
    df["team_size"] = pd.to_numeric(df["team_size"], errors="coerce")
    df["top_company"] = df["top_company"].fillna(False).astype(bool)
    return df



# ------------------------------------------------------------------ Census API
def census_get(dataset: str, params: dict, name: str) -> pd.DataFrame:
    """GET https://api.census.gov/data/{dataset} with the key from .env; cached as CSV.
    Raises with the server's message (key redacted) if the query is rejected."""
    cache = RAW / "census" / f"{name}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC:
        return _synthetic_census(name)
    if not cache.exists():
        import requests
        key = _need("CENSUS_API_KEY")
        r = requests.get(f"https://api.census.gov/data/{dataset}", params={**params, "key": key}, timeout=120)
        if r.status_code != 200 or not r.text.startswith("["):
            raise RuntimeError(f"Census {dataset}: HTTP {r.status_code}: {r.text.replace(key, '<KEY>')[:400]}")
        rows = r.json()
        pd.DataFrame(rows[1:], columns=rows[0]).to_csv(cache, index=False)
    return pd.read_csv(cache, dtype=str)


# ------------------------------------------------------------------ SEC: industry codes and annual financials
FSDS = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip"


def sec_sic(quarters=("2013q1", "2017q1", "2021q1", "2025q1")) -> pd.Series:
    """SIC code for each CIK, from the 'sub.txt' table of SEC Financial Statement Data Sets.
    Each zip is about 50-90 MB; only sub.txt is kept. Latest quarter wins for firms that appear
    in several."""
    if SYNTHETIC:
        return _synthetic_sic()
    import io, zipfile
    parts = []
    for q in quarters:
        cache = RAW / "sec" / f"fsds_sub_{q}.csv"
        if not cache.exists():
            import requests
            r = requests.get(FSDS.format(q=q), headers={"User-Agent": _need("SEC_USER_AGENT")}, timeout=600)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            sub = pd.read_csv(z.open("sub.txt"), sep="\t", usecols=["cik", "name", "sic"], dtype=str)
            sub.to_csv(cache, index=False)
            time.sleep(0.2)
        parts.append(pd.read_csv(cache, dtype={"sic": str}).assign(q=q))
    sub = pd.concat(parts).dropna(subset=["sic"])
    sub["cik"] = sub.cik.astype(int)
    return sub.sort_values("q").groupby("cik").sic.last().str.zfill(4)


REVENUE_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"]


def annual_financials(years) -> pd.DataFrame:
    """Firm-year revenue, operating income, net income and assets from SEC annual frames.
    Revenue is the largest of the common revenue tags (firms switched tags under ASC 606)."""
    out = []
    for y in years:
        rev = pd.concat([sec_frame(t, f"CY{y}") for t in REVENUE_TAGS])
        rev = rev.groupby("cik").agg(entityName=("entityName", "last"), revenue=("val", "max"))
        oi = sec_frame("OperatingIncomeLoss", f"CY{y}").groupby("cik").val.last().rename("op_income")
        ni = sec_frame("NetIncomeLoss", f"CY{y}").groupby("cik").val.last().rename("net_income")
        at = sec_frame("Assets", f"CY{y}Q4I").groupby("cik").val.last().rename("assets")
        out.append(rev.join([oi, ni, at], how="left").assign(year=y).reset_index())
    df = pd.concat(out, ignore_index=True)
    return df[df.revenue > 0]


def financial_panel(years, sic=None) -> pd.DataFrame:
    """Cleaned non-financial firm-year panel used from H2 on: numeric columns, SIC codes, revenue >= $10m,
    revenue scale errors dropped, operating margin within +/-100%, ROA within +/-100% (else NaN).
    Predecessor CIKs are re-keyed to their successors (Google -> Alphabet etc.)."""
    fin = annual_financials(years)
    for c in ["revenue", "op_income", "net_income", "assets"]:
        fin[c] = pd.to_numeric(fin[c], errors="coerce").astype(float)
    fin["year"] = fin["year"].astype(int)
    sic = sec_sic() if sic is None else sic
    fin["sic"] = fin.cik.map(sic)
    fin["cik"] = fin.cik.map(_final)
    fin["sic"] = fin.sic.fillna(fin.cik.map(sic))
    fin = fin.dropna(subset=["sic", "revenue"])
    fin = fin.sort_values("revenue").groupby(["cik", "year"], as_index=False).last()   # one row per firm-year
    s2 = fin.sic.str[:2].astype(int)
    fin = fin[~s2.between(60, 67) & (s2 < 90) & (fin.revenue >= 1e7)].copy()
    bad = (fin.revenue >= 1e12) | ((fin.revenue > 1e9) & (fin.revenue > 20 * fin.assets))
    fin = fin[~bad].copy()
    fin["op_margin"] = fin.op_income / fin.revenue
    fin["roa"] = (fin.net_income / fin.assets.where(fin.assets > 0)).replace([np.inf, -np.inf], np.nan)
    fin = fin[fin.op_margin.between(-1, 1)].copy()
    fin.loc[~fin.roa.between(-1, 1), "roa"] = np.nan
    return fin


OCF_TAGS = ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
CAPEX_TAGS = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]


def annual_tag(tag: str, years, name: str = None) -> pd.DataFrame:
    """One us-gaap duration tag from the annual frames, as cik-year rows (successor-linked)."""
    out = []
    for y in years:
        f = sec_frame(tag, f"CY{y}")
        if len(f):
            out.append(f.groupby("cik").val.last().rename(name or tag).to_frame().assign(year=y).reset_index())
    d = pd.concat(out, ignore_index=True)
    d[name or tag] = pd.to_numeric(d[name or tag], errors="coerce").astype(float)
    d["cik"] = d.cik.map(_final)
    return d.groupby(["cik", "year"], as_index=False)[name or tag].sum()


def cash_flows(years) -> pd.DataFrame:
    """Firm-year operating cash flow, capital expenditure and free cash flow (OCF - capex) from SEC
    annual frames. OCF takes the first tag a firm reports; capex the larger of the two common tags
    (Amazon, for example, reports 'productive assets'). CIKs are re-keyed to successors."""
    out = []
    for y in years:
        o = [sec_frame(t, f"CY{y}").groupby("cik").val.last().rename(t) for t in OCF_TAGS]
        c = [sec_frame(t, f"CY{y}").groupby("cik").val.last().rename(t) for t in CAPEX_TAGS]
        d = pd.concat(o + c, axis=1)
        d = d.apply(pd.to_numeric, errors="coerce").astype(float)
        cf = pd.DataFrame({"ocf": d[OCF_TAGS[0]].fillna(d[OCF_TAGS[1]]),
                           "capex": d[CAPEX_TAGS].max(axis=1).fillna(0.0)})
        out.append(cf.dropna(subset=["ocf"]).assign(year=y).rename_axis("cik").reset_index())
    cf = pd.concat(out, ignore_index=True)
    cf["cik"] = cf.cik.map(_final)
    cf = cf.groupby(["cik", "year"], as_index=False)[["ocf", "capex"]].sum()
    cf["fcf"] = cf.ocf - cf.capex
    return cf


# ------------------------------------------------------------------ regression
def fe_ols(df: pd.DataFrame, y: str, xs: list, fes: list, cluster: str, tol=1e-8, maxit=500):
    """OLS of y on xs with any number of fixed effects absorbed by alternating demeaning
    (equivalent to reghdfe), standard errors clustered by `cluster`. Returns a coefficient table."""
    d = df[[y, *xs, *fes, cluster]].dropna().copy()
    cols = [y, *xs]
    Z = d[cols].astype(float)
    if fes:
        for _ in range(maxit):
            prev = Z.copy()
            for fe in fes:
                Z = Z - Z.groupby(d[fe].values).transform("mean")
            if np.abs(Z - prev).to_numpy().max() < tol:
                break
    Y, X = Z[y].values, Z[xs].values
    if not fes:
        X = np.column_stack([np.ones(len(X)), X]); xs = ["const", *xs]
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    e = Y - X @ b
    g = d[cluster].values
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gi in pd.unique(g):
        s = X[g == gi].T @ e[g == gi]
        meat += np.outer(s, s)
    G, N, K = len(pd.unique(g)), len(Y), X.shape[1]
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((N - 1) / (N - K))
    se = np.sqrt(np.diag(V))
    from scipy import stats
    t = b / se
    p = 2 * stats.t.sf(np.abs(t), G - 1)
    return pd.DataFrame({"coef": b, "se": se, "t": t, "p": p, "n": N, "clusters": G}, index=xs)


# ------------------------------------------------------------------ FRED
def fred(series_id: str) -> pd.Series:
    """Full history of one FRED series, cached as CSV (the key is never written to disk).
    Returns an empty series if FRED does not know the id, so callers can skip it.
    For BLS CPI ids (CUUR...) that FRED does not carry, falls back to the seasonally adjusted
    FRED id (CUSR...) and then to the BLS public API."""
    s = _fred_raw(series_id)
    if s.empty and series_id.startswith("CUUR") and not SYNTHETIC:
        s = _fred_raw("CUSR" + series_id[4:])
        if s.empty:
            s = bls_series(series_id)
    return s.rename(series_id)


def bls_series(series_id: str, start: int = 1997) -> pd.Series:
    """Monthly values of one BLS series from the BLS public API (v1, no key: 10 years per request,
    25 requests per day). Cached as CSV."""
    cache = RAW / "bls" / f"{series_id}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        import requests
        rows = []
        end = pd.Timestamp.today().year
        for y0 in range(start, end + 1, 10):
            r = requests.post("https://api.bls.gov/publicAPI/v1/timeseries/data/",
                              json={"seriesid": [series_id], "startyear": str(y0), "endyear": str(min(y0 + 9, end))},
                              timeout=60)
            j = r.json()
            if j.get("status") != "REQUEST_SUCCEEDED":
                print(f"BLS {series_id}: {j.get('status')} {j.get('message')}"); break
            for d in j["Results"]["series"][0]["data"]:
                if d["period"].startswith("M") and d["period"] != "M13":
                    rows.append({"date": f"{d['year']}-{d['period'][1:]}-01", "value": d["value"]})
            time.sleep(0.5)
        pd.DataFrame(rows, columns=["date", "value"]).to_csv(cache, index=False)
    d = pd.read_csv(cache)
    if d.empty:
        return pd.Series(dtype=float, name=series_id)
    v = pd.to_numeric(d["value"], errors="coerce")
    v.index = pd.to_datetime(d["date"])
    return v.dropna().sort_index().rename(series_id)


def _fred_raw(series_id: str) -> pd.Series:
    cache = RAW / "fred" / f"{series_id}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC:
        return _synthetic_fred(series_id)
    if not cache.exists():
        import requests
        key = _need("FRED_API_KEY")
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": series_id, "api_key": key, "file_type": "json"}, timeout=60)
        if r.status_code == 400 and "does not exist" in r.text:
            cache.write_text("date,value\n")
        elif r.status_code != 200:
            raise RuntimeError(f"FRED {series_id}: HTTP {r.status_code}: {r.text.replace(key, '<KEY>')[:300]}")
        else:
            pd.DataFrame(r.json()["observations"])[["date", "value"]].to_csv(cache, index=False)
    d = pd.read_csv(cache)
    if d.empty:
        return pd.Series(dtype=float, name=series_id)
    s = pd.to_numeric(d["value"], errors="coerce")
    s.index = pd.to_datetime(d["date"])
    return s.dropna().rename(series_id)


def _synthetic_fred(sid):
    rng = np.random.default_rng(abs(hash(sid)) % 2**32)
    d = pd.date_range("1947-01-01", "2026-07-01", freq="QS" if sid in ("OPHNFB", "PNFI") or sid.endswith("Q027SBEA") else "MS")
    if sid.startswith(("RTFP", "SPDYN")):
        d = pd.date_range("1960-01-01", "2023-01-01", freq="YS")
    g = {"CUUR0000SEEE01": -0.07, "CUUR0000SERA01": -0.10, "CUUR0000SEEE02": -0.03, "CUUR0000SEED03": -0.02,
         "CUUR0000SEEB01": 0.055, "CPIMEDSL": 0.045}.get(sid, 0.03)
    t = np.arange(len(d)) / (4 if "Q" in getattr(d, "freqstr", "") else (1 if len(d) < 100 else 12))
    return pd.Series(100 * np.exp(g * t + np.cumsum(rng.normal(0, 0.005, len(d)))), index=d, name=sid)


# ------------------------------------------------------------------ prices
def yahoo_close(ticker: str, adjusted: bool = True, start: str = "2010-01-01") -> pd.Series:
    """Daily close from Yahoo Finance (yfinance). adjusted=True: dividends and splits reinvested
    (total return); adjusted=False: split-adjusted close only. Cached as CSV; empty if Yahoo has none."""
    safe = re.sub(r"[^A-Za-z0-9]", "_", ticker) + ("_tr" if adjusted else "_px") + ("" if start == "2010-01-01" else f"_from{start[:4]}")
    cache = RAW / "prices" / f"yahoo_{safe}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC:
        return _synthetic_px(ticker)
    if not cache.exists():
        import yfinance as yf
        try:
            df = yf.download(ticker, start=start, auto_adjust=adjusted, progress=False, threads=False)
        except Exception:
            df = None
        if df is None or len(df) == 0:
            pd.Series(dtype=float, name="close").to_csv(cache, index_label="date")
        else:
            c = df["Close"]
            c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
            c.rename("close").to_csv(cache, index_label="date")
    d = pd.read_csv(cache, parse_dates=["date"], index_col="date")
    return d["close"].dropna().rename(ticker) if len(d) else pd.Series(dtype=float, name=ticker)


def stooq_close(symbol: str) -> pd.Series:
    """Daily split-adjusted close from Stooq (keeps some delisted US tickers, e.g. 'twtr.us')."""
    cache = RAW / "prices" / f"stooq_{symbol.replace('.', '_')}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if SYNTHETIC:
        return _synthetic_px(symbol)
    if not cache.exists():
        import requests
        try:
            r = requests.get(f"https://stooq.com/q/d/l/?s={symbol}&i=d", timeout=60)
            ok = r.status_code == 200 and r.text.startswith("Date")
        except Exception:
            ok = False
        cache.write_text(r.text if ok else "Date,Close\n")
    d = pd.read_csv(cache)
    if d.empty:
        return pd.Series(dtype=float, name=symbol)
    return pd.Series(d["Close"].values, index=pd.to_datetime(d["Date"]), name=symbol).dropna()


def _synthetic_px(t):
    rng = np.random.default_rng(abs(hash(t)) % 2**32)
    d = pd.bdate_range("2012-01-02", "2026-11-05")
    return pd.Series(50 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, len(d)))), index=d, name=t)


# ------------------------------------------------------------------ stats helpers
def tail_fit(x, xmin=None):
    """Clauset-Shalizi-Newman (2009) maximum-likelihood power-law fit of the upper tail.
    Returns alpha (pdf exponent), its standard error, xmin and the number of tail observations.
    alpha near 2 corresponds to Zipf's law for firm size (Axtell 2001)."""
    import powerlaw
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    fit = powerlaw.Fit(x, xmin=xmin, verbose=False)
    n = int((x >= fit.xmin).sum())
    return {"alpha": fit.power_law.alpha, "se": fit.power_law.sigma, "xmin": fit.xmin, "n_tail": n,
            "fit": fit}


def compare_tails(fit):
    """Likelihood-ratio test of power law vs lognormal (positive R favours power law)."""
    R, p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    return {"R_pl_vs_lognormal": R, "p": p}


def top_share(x, k=None, frac=None):
    x = np.sort(np.asarray(x, float))[::-1]
    if frac is not None:
        k = max(1, int(round(frac * len(x))))
    return x[:k].sum() / x.sum()


def ccdf(x):
    x = np.sort(np.asarray(x, float))
    return x, 1 - np.arange(len(x)) / len(x)


# ------------------------------------------------------------------ output
def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.8, "legend.frameon": False,
    })


def save_fig(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")


def save_table(df, name, **kw):
    df.to_csv(TAB / f"{name}.csv", **kw)


# ------------------------------------------------------------------ synthetic (testing only)
_SYN = {}


def _synthetic_frame(tag, period):
    """Pareto-distributed values for ~9,000 filers with persistent identities across years.
    Annual periods ('CY2014') and Q4 instants are treated as the Q4 cohort plus everyone."""
    if "pop" not in _SYN:
        rng = np.random.default_rng(1)
        n = 9000
        _SYN["pop"] = pd.DataFrame({"cik": np.arange(1, n + 1),
                                    "entityName": [f"FIRM {i}" for i in range(1, n + 1)],
                                    "born": rng.integers(2000, 2025, n),
                                    "dies": rng.integers(2005, 2040, n),
                                    "q": rng.integers(1, 5, n),
                                    "size0": 2e7 * (1 + rng.pareto(1.05, n)),
                                    "g": rng.normal(0.06, 0.35, n)})
    pop = _SYN["pop"]
    y = int(period[2:6]); q = int(period[7]) if len(period) > 7 else 0
    rng = np.random.default_rng(y * 10 + q + len(tag))
    alive = pop[(pop.born <= y) & (pop.dies > y) & ((pop.q == q) if (q and tag == "EntityPublicFloat") else True)]
    t = y - alive.born
    val = alive.size0 * np.exp(alive.g * t + rng.normal(0, 0.2, len(alive)))
    if tag != "EntityPublicFloat":
        sic = _synthetic_sic().reindex(alive.cik).values
        conc = pd.Series(sic).map(lambda c: (int(c) % 7) / 7).values            # industry "concentration" driver
        if tag == "OperatingIncomeLoss":
            val = val * 0.6 * (0.03 + 0.15 * conc + rng.normal(0, 0.08, len(alive)))
        elif tag == "NetIncomeLoss":
            val = val * 0.6 * (0.02 + 0.10 * conc + rng.normal(0, 0.08, len(alive)))
        elif tag == "Assets":
            val = val * 1.3
        elif tag.startswith("SellingAndMarketing"):
            val = val * 0.6 * np.abs(0.18 + rng.normal(0, 0.10, len(alive)))
        elif tag.startswith("ResearchAndDevelopment"):
            val = val * 0.6 * np.abs(0.12 + rng.normal(0, 0.08, len(alive)))
        elif tag.startswith("SellingGeneral"):
            val = val * 0.6 * np.abs(0.25 + rng.normal(0, 0.10, len(alive)))
        elif tag.startswith("NetCashProvided"):
            val = val * 0.6 * (0.10 + 0.12 * conc + rng.normal(0, 0.06, len(alive)))
        elif tag.startswith("PaymentsToAcquire"):
            val = val * 0.6 * np.abs(0.04 + rng.normal(0, 0.02, len(alive)))
        else:
            val = val * 0.6
    end = pd.Timestamp(year=y, month=3 * max(q, 1), day=28)
    return pd.DataFrame({"accn": "x", "cik": alive.cik, "entityName": alive.entityName,
                         "loc": "US", "end": end, "val": val.values})


def _synthetic_sic():
    if "sic" not in _SYN:
        _synthetic_frame("EntityPublicFloat", "CY2014Q1I")
        rng = np.random.default_rng(5)
        codes = [f"{c:04d}" for c in rng.choice(np.arange(1000, 8000, 10), 60, replace=False)] + ["4512", "7370"]
        _SYN["sic"] = pd.Series(rng.choice(codes, len(_SYN["pop"])), index=_SYN["pop"].cik)
    return _SYN["sic"]


def _synthetic_census(name):
    r0 = np.random.default_rng(0)
    naics = sorted({f"3{r0.integers(11, 40)}{r0.integers(100, 999)}" for _ in range(300)})
    rng = np.random.default_rng(len(name))
    if name.startswith("concen"):
        rows = []
        for n in naics:
            hhi = rng.uniform(100, 3000)
            for code, lab in [("001", "All firms"), ("002", "4 largest companies"), ("003", "8 largest companies")]:
                rows.append({"NAICS2017": n, "NAICS2022": n, "CONCENFI": code, "VAL_PCT": "" if code == "001" else f"{min(99, hhi/40 + 10 * int(code)):.1f}",
                             "HHI": f"{hhi:.1f}" if code == "001" else "", "RCPTOT": str(int(rng.uniform(1e5, 1e8))), "FIRM": str(rng.integers(5, 2000))})
        return pd.DataFrame(rows)
    rows = []
    for n in naics:
        for yr in range(2012, 2023):
            rcpt = rng.uniform(1e5, 1e8); va = rcpt * rng.uniform(0.3, 0.6)
            rows.append({"NAICS2017": n, "YEAR": str(yr), "RCPTOT": str(int(rcpt)), "VALADD": str(int(va)),
                         "PAYANN": str(int(va * rng.uniform(0.3, 0.6))), "CSTMTOT": str(int(rcpt - va))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ H8: occupations (BLS OEWS, O*NET, AI exposure)
def _ua() -> str:
    """Browser-style User-Agent with the contact from SEC_USER_AGENT (BLS refuses anonymous scripts)."""
    contact = load_env().get("SEC_USER_AGENT") or os.environ.get("SEC_USER_AGENT") or "research script"
    return f"Mozilla/5.0 (compatible; academic research; {contact})"


def _download(url: str, path: Path, what: str) -> bool:
    """Download url to path once. Returns False (and prints a manual-download hint) on failure."""
    if path.exists() and path.stat().st_size > 0:
        return True
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, headers={"User-Agent": _ua(), "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"},
                         timeout=120)
    except Exception as e:
        print(f"{what}: download failed ({type(e).__name__}).")
        r = None
    if r is not None and r.status_code == 200 and not r.content[:200].lstrip().lower().startswith((b"<!doctype", b"<html")):
        path.write_bytes(r.content)
        return True
    code = r.status_code if r is not None else "no response"
    if code in (200, 404):
        print(f"{what}: not published yet (skipped)."); return False
    print(f"{what}: could not download (HTTP {code}). If this keeps happening, open\n  {url}\n"
          f"in a browser and save the file as\n  {path}")
    return False


def oews_national(year: int) -> pd.DataFrame:
    """BLS OEWS national estimates (May of `year`), detailed occupations only:
    occ_code, title, emp, wage (annual mean wage). Empty frame if the year is not published."""
    if SYNTHETIC:
        return _synthetic_h8()["oews"].get(year, pd.DataFrame())
    cache = RAW / "oews" / f"national_{year}.csv"
    if not cache.exists():
        import zipfile
        yy = str(year)[2:]
        z = RAW / "oews" / f"oesm{yy}nat.zip"
        if not _download(f"https://www.bls.gov/oes/special-requests/oesm{yy}nat.zip", z, f"OEWS {year}"):
            return pd.DataFrame()
        if not zipfile.is_zipfile(z):
            z.unlink(); print(f"OEWS {year}: the download was not a zip file (deleted); try again."); return pd.DataFrame()
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if re.search(r"nat.*\.xlsx?$", n.lower()) and "field" not in n.lower()]
            if not names:
                raise RuntimeError(f"OEWS {year}: no national spreadsheet in {z.name}: {zf.namelist()}")
            with zf.open(sorted(names, key=len)[0]) as fh:
                raw = pd.read_excel(fh, header=None, dtype=str)
        hdr = next(i for i in range(min(15, len(raw))) if raw.iloc[i].astype(str).str.upper().eq("OCC_CODE").any())
        df = raw.iloc[hdr + 1:].copy()
        df.columns = raw.iloc[hdr].astype(str).str.upper().str.strip()
        grp = "O_GROUP" if "O_GROUP" in df.columns else "OCC_GROUP"
        df = df[df[grp].astype(str).str.lower().str.strip() == "detailed"]
        out = pd.DataFrame({"occ_code": df.OCC_CODE.str.strip(), "title": df.OCC_TITLE.str.strip(),
                            "emp": pd.to_numeric(df.TOT_EMP, errors="coerce"),
                            "wage": pd.to_numeric(df.A_MEAN, errors="coerce")})
        out.to_csv(cache, index=False)
    return pd.read_csv(cache, dtype={"occ_code": str})


def latest_oews_year(candidates=(2026, 2025, 2024, 2023)) -> int:
    for y in candidates:
        if not oews_national(y).empty:
            return y
    raise RuntimeError("No recent OEWS year could be loaded")


def soc_crosswalk() -> pd.DataFrame:
    """BLS 2010-to-2018 SOC crosswalk: columns soc2010, soc2018."""
    if SYNTHETIC:
        return _synthetic_h8()["xwalk"]
    cache = RAW / "oews" / "soc_2010_2018.csv"
    if not cache.exists():
        f = RAW / "oews" / "soc_2010_to_2018_crosswalk.xlsx"
        if not _download("https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx", f, "SOC crosswalk"):
            return pd.DataFrame(columns=["soc2010", "soc2018"])
        raw = pd.read_excel(f, header=None, dtype=str)
        hdr = next(i for i in range(30) if raw.iloc[i].astype(str).str.contains("2010 SOC Code", case=False).any())
        cols = raw.iloc[hdr].astype(str).str.strip()
        c10 = next(i for i, c in enumerate(cols) if re.search("2010 SOC Code", c, re.I))
        c18 = next(i for i, c in enumerate(cols) if re.search("2018 SOC Code", c, re.I))
        x = raw.iloc[hdr + 1:, [c10, c18]].dropna()
        x.columns = ["soc2010", "soc2018"]
        x = x[x.soc2010.str.match(r"^\d\d-\d{4}$") & x.soc2018.str.match(r"^\d\d-\d{4}$")]
        x.to_csv(cache, index=False)
    return pd.read_csv(cache, dtype=str)


ONET_VERSIONS = ("30_1", "30_0", "29_3", "29_2", "29_1", "29_0", "28_3")


def onet_scores() -> pd.DataFrame:
    """O*NET task measures by 6-digit SOC 2018 code (mean over O*NET-SOC detail codes):
    computer = importance of 'Working with Computers' (1-5),
    automation = 'Degree of Automation' (1-5), repetitive = 'Importance of Repeating Same Tasks' (1-5)."""
    if SYNTHETIC:
        return _synthetic_h8()["onet"]
    cache = RAW / "onet" / "onet_scores.csv"
    if not cache.exists():
        got = {}
        for fname in ("Work Activities", "Work Context"):
            for v in ONET_VERSIONS:
                p = RAW / "onet" / f"{fname} {v}.txt"
                url = f"https://www.onetcenter.org/dl_files/database/db_{v}_text/{fname.replace(' ', '%20')}.txt"
                if p.exists() or _download(url, p, f"O*NET {v} {fname}"):
                    got[fname] = pd.read_csv(p, sep="\t", dtype=str); break
        if len(got) < 2:
            raise RuntimeError("O*NET files could not be downloaded; see the messages above")
        rows = []
        for fname, pat, scale, name in [("Work Activities", r"with computers", "IM", "computer"),
                                        ("Work Context", r"degree of automation", "CX", "automation"),
                                        ("Work Context", r"repeating same tasks", "CX", "repetitive")]:
            d = got[fname]
            d = d[d["Element Name"].str.contains(pat, case=False, regex=True) & (d["Scale ID"] == scale)]
            rows.append(pd.DataFrame({"soc": d["O*NET-SOC Code"].str[:7], "var": name,
                                      "value": pd.to_numeric(d["Data Value"], errors="coerce")}))
        s = pd.concat(rows).groupby(["soc", "var"]).value.mean().unstack()
        s.to_csv(cache)
    return pd.read_csv(cache, dtype={"soc": str}).set_index("soc")


def aioe_scores() -> pd.Series:
    """Felten, Raj and Seamans (2021) AI Occupational Exposure by SOC 2010 code (GitHub data appendix)."""
    if SYNTHETIC:
        return _synthetic_h8()["aioe"]
    f = RAW / "aioe" / "AIOE_DataAppendix.xlsx"
    if not _download("https://raw.githubusercontent.com/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx", f, "AIOE"):
        return pd.Series(dtype=float)
    d = pd.read_excel(f, sheet_name="Appendix A", dtype={"SOC Code": str})
    return d.set_index("SOC Code")["AIOE"].astype(float)


def soc_groups(codes2010, codes2018, xwalk: pd.DataFrame) -> tuple[dict, dict]:
    """Assign SOC 2010 and SOC 2018 codes to stable groups: connected components of the crosswalk.
    Codes absent from the crosswalk that end in 0 (OEWS aggregates such as 29-1060) join every
    code sharing their first five digits. Returns two dicts code -> group id."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for a, b in zip(xwalk.soc2010, xwalk.soc2018):
        union("A" + a, "B" + b)
    side_codes = {"A": set(xwalk.soc2010), "B": set(xwalk.soc2018)}
    for side, codes in (("A", codes2010), ("B", codes2018)):
        for c in set(codes):
            if c in side_codes[side]:
                continue
            kids = [k for k in side_codes[side] if c.endswith("0") and k[:6] == c[:6]]
            for k in kids:
                union(side + c, side + k)
            if not kids:
                find(side + c)                       # stays on its own (matched only if the code is unchanged)
    # unchanged codes not in the crosswalk: link A:x with B:x
    for c in set(codes2010) & set(codes2018):
        if c not in side_codes["A"] and c not in side_codes["B"]:
            union("A" + c, "B" + c)
    g10 = {c: find("A" + c) for c in set(codes2010)}
    g18 = {c: find("B" + c) for c in set(codes2018)}
    return g10, g18


def wls(y, X, w, names):
    """Weighted least squares with heteroskedasticity-robust (HC1) standard errors."""
    from scipy import stats
    X = np.column_stack([np.ones(len(y)), X]); w = np.asarray(w, float) / np.mean(w)
    sw = np.sqrt(w); Xw, yw = X * sw[:, None], np.asarray(y) * sw
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    b = XtX_inv @ Xw.T @ yw
    e = yw - Xw @ b
    n, k = Xw.shape
    V = XtX_inv @ (Xw.T * e**2) @ Xw @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(V))
    p = 2 * stats.t.sf(np.abs(b / se), n - k)
    r2 = 1 - (e @ e) / (((yw - np.average(y, weights=w) * sw) ** 2).sum())
    return pd.DataFrame({"x": ["const", *names], "coef": b, "se": se, "p": p, "n": n, "r2": r2})


def _synthetic_h8():
    if "h8" in _SYN:
        return _SYN["h8"]
    rng = np.random.default_rng(8)
    n = 400
    c18 = sorted({f"{rng.integers(11, 54):02d}-{rng.integers(1000, 9999):04d}" for _ in range(n * 2)})[:n]
    c18 = [c[:-1] + ("1" if c[-1] == "0" else c[-1]) for c in c18]
    c18 = list(dict.fromkeys(c18))
    c10 = [c[:3] + str((int(c[3]) + 1) % 10) + c[4:] if i % 5 == 0 else c for i, c in enumerate(c18)]
    xw = pd.DataFrame({"soc2010": c10, "soc2018": c18})
    xw = pd.concat([xw, pd.DataFrame({"soc2010": c10[:10], "soc2018": c18[10:20]})], ignore_index=True)
    comp = rng.uniform(1.5, 4.9, len(c18)); auto = rng.uniform(1.2, 3.5, len(c18)); rep = rng.uniform(2, 4.5, len(c18))
    onet = pd.DataFrame({"computer": comp, "automation": auto, "repetitive": rep}, index=pd.Index(c18, name="soc"))
    aioe = pd.Series(0.6 * (comp - 3) + rng.normal(0, 0.6, len(c10)), index=c10)
    e14 = np.exp(rng.normal(10.5, 1.3, len(c18))); w14 = 30000 + 18000 * comp + rng.normal(0, 8000, len(c18))
    oews = {}
    for y in (2014, 2022, 2023, 2024, 2025):
        t = y - 2014
        g = 0.006 * (comp - 3) - 0.01 * (rep - 3) + (0.002 * (aioe.values - 0) * max(0, y - 2022))
        emp = e14 * np.exp(t * (0.008 + g) + rng.normal(0, 0.03, len(c18)))
        wage = w14 * np.exp(t * (0.028 + 0.004 * (comp - 3)) + rng.normal(0, 0.02, len(c18)))
        codes = c10 if y == 2014 else c18
        oews[y] = pd.DataFrame({"occ_code": codes, "title": [f"Occupation {c}" for c in c18], "emp": emp.round(), "wage": wage.round()})
    _SYN["h8"] = {"xwalk": xw, "onet": onet, "aioe": aioe, "oews": oews}
    return _SYN["h8"]
