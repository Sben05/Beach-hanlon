"""Transparent Python reanalysis of the England and Wales district panel."""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

MARRIAGE = [
    "mar_rate_73_77", "shrmar_registrar_73_77", "shrmar_catholic_73_77",
    "shrmar_minors_73_77", "shrmar_illit_73_77", "shrmar_first_73_77",
]
DISTRICT = [
    "density_71", "total_mr_71_80", "under5_mr_71_80",
    "illeg_birth_share_73_77", "shr_manuf", "shr_agriculture",
    "u30_fert_shr_1871",
]
CONTROL_NAMES = ["Region effects", "+ District and marriage", "+ Existing papers", "+ Lagged fertility"]


def _read_frame(stem):
    """Use compact CSV.gz everywhere; support legacy Parquet if available."""
    csv = DATA / f"{stem}.csv.gz"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.read_parquet(DATA / f"{stem}.parquet")


def load_data():
    panel = _read_frame("district_periods")
    annual = _read_frame("district_annual")
    monthly = _read_frame("newspaper_monthly_queries")
    yearly = _read_frame("newspaper_yearly_queries")
    return panel, annual, monthly, yearly


def base_differences(panel):
    pre = panel.loc[panel.year == 1873].set_index("district")
    post = panel.loc[panel.year == 1878].set_index("district").copy()
    post["dy"] = post.birth_rate - pre.birth_rate
    post["dlag"] = post.birth_rate_lagged - pre.birth_rate_lagged
    return post.reset_index()


def _design(frame, exposure, controls, *, paper_column=None, outcome="dy", weighted=True):
    cols = [exposure]
    if controls >= 1:
        cols += MARRIAGE + DISTRICT
    if controls >= 2 and paper_column:
        cols.append(paper_column)
    if controls >= 3:
        cols.append("dlag")
    cols = list(dict.fromkeys(cols))
    z = frame[[outcome, "region", "total_pop_1871", *cols]].dropna().copy()
    reg = pd.get_dummies(z.region, prefix="region", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([z[cols].astype(float), reg], axis=1))
    weights = z.total_pop_1871 if weighted else np.ones(len(z))
    fit = sm.WLS(z[outcome].astype(float), X, weights=weights).fit(cov_type="HC1")
    return fit, z


def _result(fit, z, exposure):
    ci = fit.conf_int().loc[exposure]
    return {
        "beta": float(fit.params[exposure]),
        "se": float(fit.bse[exposure]),
        "p": float(fit.pvalues[exposure]),
        "lo": float(ci.iloc[0]),
        "hi": float(ci.iloc[1]),
        "n": int(len(z)),
    }


def select_sample(base, window=3, omit_1877=False):
    q = base.copy()
    early_years = range(1878 - window, 1878)
    late_years = range(1878, 1878 + window)
    q["early"] = q[[f"new_papers_{y}" for y in early_years]].sum(axis=1)
    q["late"] = q[[f"new_papers_{y}" for y in late_years]].sum(axis=1)
    q["prior_papers"] = q.established_papers + q[
        [f"new_papers_{y}" for y in range(1867, 1878 - window)]
    ].sum(axis=1)
    q = q[(q.early > 0) | ((q.early == 0) & (q.late > 0))]
    if omit_1877:
        q = q[q.new_papers_1877 == 0]
    return q.copy()


def estimate(base, window=3, controls=3, weighted=True, omit_1877=False):
    q = select_sample(base, window, omit_1877)
    fit, z = _design(q, "early", controls, paper_column="prior_papers", weighted=weighted)
    out = _result(fit, z, "early")
    out.update({
        "window": window, "controls": controls, "weighted": weighted,
        "omit_1877": omit_1877, "treated": int((q.early > 0).sum()),
        "comparison": int((q.early == 0).sum()),
    })
    return out


def specification_atlas(base, omit_1877=False):
    rows = []
    for weighted in (True, False):
        for controls in range(4):
            for window in (2, 3, 4, 5):
                rows.append(estimate(base, window, controls, weighted, omit_1877))
    return pd.DataFrame(rows)


def balance(base, window=3, omit_1877=False):
    q = select_sample(base, window, omit_1877)
    labels = {
        "mar_rate_73_77": "Marriage rate", "shrmar_catholic_73_77": "Catholic marriages",
        "shrmar_illit_73_77": "Illiterate marrying parties", "density_71": "Population density",
        "shr_manuf": "Manufacturing share", "shr_agriculture": "Agriculture share",
        "under5_mr_71_80": "Under-5 mortality", "illeg_birth_share_73_77": "Illegitimate birth share",
        "birth_rate_lagged": "Prior birth rate", "prior_papers": "Established newspapers",
    }
    rows = []
    a, b = q[q.early > 0], q[q.early == 0]
    for col, label in labels.items():
        aa, bb = a[col].dropna(), b[col].dropna()
        sd = q[col].std()
        gap = (aa.mean() - bb.mean()) / sd if sd and np.isfinite(sd) else np.nan
        rows.append({"Variable": label, "Standardized gap": gap,
                     "Earlier": aa.mean(), "Later": bb.mean()})
    return pd.DataFrame(rows)


def placebo_years(base, annual, window=3):
    """Exploratory break-year scan with newly constructed five-year birth rates."""
    facts = base.drop_duplicates("district").set_index("district")
    rows = []
    for focal in range(1872, 1883):
        periods = []
        for label, start, end in [
            ("lag", focal - 9, focal - 5),
            ("pre", focal - 4, focal),
            ("post", focal + 1, focal + 5),
        ]:
            p = annual[annual.year.between(start, end)].groupby("district").agg(
                births=("births", "sum"), fertile_pop=("fertile_pop", "sum"),
                years=("year", "nunique"),
            )
            p[f"rate_{label}"] = 1000 * p.births / p.fertile_pop
            p.loc[p.years < 5, f"rate_{label}"] = np.nan
            periods.append(p[[f"rate_{label}"]])
        q = facts.join(periods, how="left")
        q["dy"] = q.rate_post - q.rate_pre
        q["dlag"] = q.rate_pre - q.rate_lag
        q["early"] = q[[f"new_papers_{y}" for y in range(focal - window + 1, focal + 1)]].sum(axis=1)
        q["late"] = q[[f"new_papers_{y}" for y in range(focal + 1, focal + window + 1)]].sum(axis=1)
        q["prior_papers"] = q.established_papers + q[
            [f"new_papers_{y}" for y in range(1867, focal - window + 1)]
        ].sum(axis=1)
        q = q[(q.early > 0) | ((q.early == 0) & (q.late > 0))]
        try:
            fit, z = _design(q, "early", 3, paper_column="prior_papers")
            result = _result(fit, z, "early")
        except (ValueError, np.linalg.LinAlgError, KeyError):
            continue
        result["focal"] = focal
        rows.append(result)
    return pd.DataFrame(rows)


def first_stage(base):
    q = select_sample(base, 3)
    fit, z = _design(q, "early", 3, paper_column="prior_papers", outcome="articles_own")
    return _result(fit, z, "early")
