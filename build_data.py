"""Make small, portable app datasets from Beach and Hanlon's Stata files."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

MARRIAGE = [
    "mar_rate_73_77", "shrmar_registrar_73_77", "shrmar_catholic_73_77",
    "shrmar_minors_73_77", "shrmar_illit_73_77", "shrmar_first_73_77",
]
DISTRICT = [
    "density_71", "total_mr_71_80", "under5_mr_71_80",
    "illeg_birth_share_73_77", "shr_manuf", "shr_agriculture",
    "u30_fert_shr_1871",
]
PAPERS = [f"new_papers_{year}" for year in range(1867, 1888)]


def main():
    source = pd.read_stata(RAW / "Temp" / "EW_DD_cleaned.dta", convert_categoricals=False)
    cols = [
        "district", "period", "year", "dist_code", "region", "birth_rate",
        "birth_rate_lagged", "total_pop_1871", "lat", "lon", "articles_own",
        "established_papers", *MARRIAGE, *DISTRICT, *PAPERS,
    ]
    source[cols].to_parquet(OUT / "district_periods.parquet", index=False)

    births = pd.read_stata(
        RAW / "Raw_Data/England_and_Wales/district_births_data_standardized.dta",
        convert_categoricals=False,
    )[["district", "year", "births"]]
    pop = pd.read_stata(
        RAW / "Raw_Data/England_and_Wales/district_pop_by_age_combined.dta",
        convert_categoricals=False,
    )
    female_age_cols = [f"age_{age}_F" for age in range(15, 50, 5)]
    pop["fertile_pop"] = pop[female_age_cols].sum(axis=1, min_count=7)
    pop = pop[["district", "year", "fertile_pop"]]

    districts = sorted(source.district.unique())
    grid = pd.MultiIndex.from_product(
        [districts, range(1851, 1892)], names=["district", "year"]
    ).to_frame(index=False)
    grid["year"] = grid.year.astype(float)
    annual = grid.merge(pop, on=["district", "year"], how="left")
    annual = annual.sort_values(["district", "year"])
    annual["fertile_pop"] = annual.groupby("district")["fertile_pop"].transform(
        lambda s: s.interpolate(limit_area="inside")
    )
    annual = annual.merge(births, on=["district", "year"], how="left")
    annual["birth_rate"] = 1000 * annual.births / annual.fertile_pop
    annual = annual.dropna(subset=["birth_rate", "fertile_pop"])
    annual["year"] = annual.year.astype(int)
    annual.to_parquet(OUT / "district_annual.parquet", index=False)

    news_dir = RAW / "Raw_Data/England_and_Wales"
    for name in ["Newspaper_Monthly_Queries", "Newspaper_Yearly_Queries"]:
        pd.read_stata(news_dir / f"{name}.dta", convert_categoricals=False).to_parquet(
            OUT / f"{name.lower()}.parquet", index=False
        )
    print(f"Wrote {len(source)} district-period rows and {len(annual)} annual rows")


if __name__ == "__main__":
    main()
