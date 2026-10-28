"""Check key reconstructed estimates against the authors' published results."""

from analysis import base_differences, estimate, first_stage, load_data, placebo_years, specification_atlas
from referee import influence_audit, influence_context, direct_channel_sensitivity


def main():
    panel, annual, _, _ = load_data()
    base = base_differences(panel)
    expected = [-4.544, -2.962, -2.598, -2.673]
    for controls, published in enumerate(expected):
        result = estimate(base, 3, controls)
        assert result["n"] == 98
        assert abs(result["beta"] - published) < 0.001, (controls, result)
    stage = first_stage(base)
    assert abs(stage["beta"] - 9.756) < 0.001
    atlas = specification_atlas(base)
    assert len(atlas) == 32
    assert (atlas.beta < 0).all()
    placebo = placebo_years(base, annual)
    assert len(placebo) == 11
    assert placebo.loc[placebo.beta.idxmin(), "focal"] == 1877
    audit = influence_audit(base, random_trials=30)
    assert len(audit['rows']) == 98
    assert len(audit['path']) == 6
    assert abs(audit['baseline'] - (-2.673)) < 0.001
    assert abs(audit['path'][-1]['beta'] - (-0.6009522423782183)) < 0.000001
    assert audit['path'][-1]['beta'] > audit['baseline']
    context = influence_context(audit, annual)
    selected = context.loc[context.selection_order.notna()].sort_values('selection_order')
    assert len(context) == 98 and len(selected) == 5
    assert abs(context.weight_share_pct.sum() - 100) < 1e-8
    assert selected[['1871_73', '1874_76', 'pre_change']].notna().all().all()
    assert context.wls_leverage.between(0, 1).all()
    assert selected.district.tolist() == [step['district'] for step in audit['path'][1:]]
    assert all(abs(row.delta - (row.beta_without - audit['baseline'])) < 1e-10
               for row in selected.itertuples())
    without_five = estimate(base.loc[~base.district.isin(selected.district)], 3, 3)
    assert abs(without_five['beta'] - audit['path'][-1]['beta']) < 1e-8
    unweighted_full = estimate(base, 3, 3, False)
    unweighted_without_five = estimate(
        base.loc[~base.district.isin(selected.district)], 3, 3, False
    )
    assert unweighted_without_five['beta'] - unweighted_full['beta'] > 1
    assert abs(direct_channel_sensitivity(-2.673)['adjusted_per_opening']) < 1e-10
    assert abs(direct_channel_sensitivity(-0.958)['ci_hi']) < 1e-10
    print("Verified published point estimates, first stage, 32-spec grid, placebo scan, ")
    print("98 leave-one-out refits, five-district context, five-step sensitivity path, and IV thresholds.")


if __name__ == "__main__":
    main()
