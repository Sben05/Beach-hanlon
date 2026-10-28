"""Supplemental *exploratory* sensitivity diagnostics for Beach & Hanlon (2023).

All observed district results use authors' archival reproduction dataset.
The direct newspaper channel is a user-specified hypothetical, NOT estimated.
"""
import numpy as np
import pandas as pd
from analysis import MARRIAGE, DISTRICT, select_sample

PUBLISHED_REDUCED_FORM = -2.673
PUBLISHED_REDUCED_FORM_SE = 0.875
PUBLISHED_FIRST_STAGE = 9.756
PUBLISHED_IV = -0.274
PUBLISHED_IV_SE = 0.138


def direct_channel_sensitivity(gamma, beta=PUBLISHED_REDUCED_FORM,
                               se=PUBLISHED_REDUCED_FORM_SE,
                               first_stage=PUBLISHED_FIRST_STAGE):
    """Condition on a hypothetical, known, direct newspaper effect gamma.

    gamma is per early opening, in annual births/1000 women.
    Confidence interval is conditional on FIXED gamma and inherited published
    reduced form SE, NOT a valid unconditional interval when gamma is unknown.
    """
    adjusted = beta - gamma
    return {
        'gamma': float(gamma), 'baseline': beta, 'adjusted_per_opening': adjusted,
        'adjusted_per_article': adjusted/first_stage,
        'ci_lo': adjusted - 1.96*se, 'ci_hi': adjusted + 1.96*se,
        'gamma_ci_boundary': beta + 1.96*se,
        'gamma_point_boundary': beta,
        'category': 'CI excludes zero' if adjusted + 1.96*se < 0 else
                    'CI above zero' if adjusted - 1.96*se > 0 else 'CI includes zero',
    }


def _full_matrix(base):
    """98-district published-window sample / WLS point-estimate design."""
    q = select_sample(base, 3).copy().reset_index(drop=True)
    cols = ['early', *MARRIAGE, *DISTRICT, 'prior_papers', 'dlag']
    needed = ['district', 'region', 'total_pop_1871', 'dy', 'articles_own', *cols]
    z = q[needed].dropna().copy().reset_index(drop=True)
    reg = pd.get_dummies(z.region, prefix='region', drop_first=True, dtype=float)
    xx = pd.concat([pd.Series(1.0, index=z.index, name='constant'),z[cols].astype(float),reg],axis=1)
    X=xx.to_numpy(dtype=float)
    y=z.dy.to_numpy(dtype=float)
    w=z.total_pop_1871.to_numpy(dtype=float)
    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:,None]
    yw = y * sqrt_w
    return z, Xw, yw, list(xx.columns).index('early')


def _beta(xw,yw,idx,mask):
    coefs = np.linalg.lstsq(xw[mask],yw[mask],rcond=None)[0]
    return float(coefs[idx])


def influence_audit(base, kmax=5, random_trials=300, seed=1877):
    """Greedy deletion stress test. NOT a causal estimate or statistical test.

    Greedy algorithm chooses district whose removal makes beta least negative,
    one district at a time. Random comparators of the same deletion count show
    typical descriptive sensitivity, with no inferential interpretation.
    """
    z, X, y, idx = _full_matrix(base)
    n=len(z)
    if n != 98:
        raise ValueError(f'Expected 98-district sample, got {n}')
    full=_beta(X,y,idx,np.ones(n,dtype=bool))
    hat=np.sum(X*np.linalg.pinv(X).T,axis=1)
    rows=[]
    for i,row in z.iterrows():
        mask=np.ones(n,dtype=bool);mask[i]=False
        candidate=_beta(X,y,idx,mask)
        rows.append({
            'district': str(row.district), 'region': str(row.region),
            'group': 'Earlier' if row.early>0 else 'Later',
            'early_openings': float(row.early), 'articles': float(row.articles_own),
            'population': int(row.total_pop_1871), 'beta_without': candidate,
            'delta': candidate-full, 'wls_leverage': float(hat[i]),
        })
    loo=pd.DataFrame(rows).sort_values('delta',ascending=False).reset_index(drop=True)

    left=np.ones(n,dtype=bool)
    path=[{'removed':0,'district':'—','beta':full,'delta':0.0,'n':n}]
    for k in range(1,kmax+1):
        candidate_choice=None
        for i in np.where(left)[0]:
            trial=left.copy();trial[i]=False
            # keep robustly sized comparison groups
            if (np.sum((z.early.values>0)&trial)<30 or
                np.sum((z.early.values==0)&trial)<20):
                continue
            b=_beta(X,y,idx,trial)
            if candidate_choice is None or b > candidate_choice[0]:
                candidate_choice=(b,i)
        if candidate_choice is None: break
        b,i=candidate_choice
        left[i]=False
        path.append({'removed':k,'district':str(z.iloc[i].district),
                     'beta':b,'delta':b-full,'n':n-k})

    # Descriptive random deletion reference distributions; reproducible seed.
    rng=np.random.default_rng(seed)
    random_ref={}
    for k in range(1,kmax+1):
        sims=[]
        for _ in range(random_trials):
            rem=rng.choice(n,size=k,replace=False)
            trial=np.ones(n,dtype=bool);trial[rem]=False
            sims.append(_beta(X,y,idx,trial))
        random_ref[k]={'min':float(np.min(sims)),
                       'q05':float(np.percentile(sims,5)),
                       'median':float(np.median(sims)),
                       'q95':float(np.percentile(sims,95)),
                       'max':float(np.max(sims)),
                       'values':[round(float(v),5) for v in sims]}
    return {'baseline':full, 'rows':loo.to_dict('records'),
            'path':path, 'random':random_ref}


def influence_context(audit, annual):
    """Pre-trial fertility and regression context for every audited district."""
    rows = pd.DataFrame(audit['rows'])
    rows['weight_share_pct'] = 100 * rows.population / rows.population.sum()
    selected = {step['district']: step['removed'] for step in audit['path'][1:]}
    rows['selection_order'] = rows.district.map(selected)

    pre = annual.loc[annual.year.between(1871, 1876),
                     ['district', 'year', 'birth_rate']].copy()
    pre['period'] = np.where(pre.year <= 1873, '1871_73', '1874_76')
    grouped = pre.groupby(['district', 'period']).birth_rate
    rates = grouped.mean().unstack()
    counts = grouped.count().unstack()
    rates = rates.where(counts == 3)

    rows = rows.join(rates, on='district')
    rows['pre_change'] = rows['1874_76'] - rows['1871_73']
    return rows
