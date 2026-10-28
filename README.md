# Sensitivity of the 1877 newspaper estimates

Small Streamlit research note based on the England and Wales district data accompanying Beach and Hanlon (2023), *Culture and the Historical Fertility Transition*.

## Question

How sensitive is the estimated newspaper-exposure effect to choices about opening windows, regression controls, district weights, influential observations, and possible alternative effects of newspaper openings?

## Main results

- Preferred Table 4 coefficient reproduced: -2.673 annual births per 1,000 women for an additional early newspaper opening.
- The paper's preferred district-clustered SE is 0.875; the first-difference reproduction's HC1 SE is 0.916. The app displays both uncertainty calculations.
- All 32 model choices in the included grid yield negative point estimates; 18 pointwise HC1 intervals exclude zero.
- Excluding districts with openings in 1877 yields -2.636 (68 rather than 98 districts).
- Deliberately selecting and removing five districts to weaken the estimate yields -0.601; this does **not** justify excluding them.
- The district audit reports each selected district's 1871 population share, pre-1877 fertility rates, weighted regression leverage, and individual leave-one-out coefficient. An unweighted re-estimation for the same five omissions is available in the app.
- Under an illustrative additive IV sensitivity model, a direct unrelated newspaper effect of about -0.958 per opening is sufficient for a conditional 95% interval to touch zero. This effect is **not measured**.

The project is an examination of sensitivity, not a causal re-estimation of the trial's contribution to nationwide fertility change. In particular, the adversarial district-removal procedure is a stress test rather than a recommended specification.

The opening-window exercise overlaps with the paper's Appendix Table 25, and the break-year scan is related to its Figure 6. Those sections are presented as reproduction and sensitivity checks, not new findings.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Data summaries are bundled as compressed CSVs, so no original Stata files are necessary to launch the app. Scripts for verifying coefficient reproduction (`verify.py`) are included. `build_data.py` needs the original Stata input files from the authors' archive to rebuild the compact CSV tables from scratch.

## Streamlit Community Cloud

Put this folder's contents at the root of a GitHub repository. In Streamlit Community Cloud, create an app from that repository, choose the branch containing these files, and set the entrypoint to `app.py`. The bundled `data/*.csv.gz`, `.streamlit/config.toml`, and `requirements.txt` are needed for deployment. No secrets are required.

## Files

- `app.py`: paper-style research interface and interactive analysis
- `analysis.py`: first-difference regressions, specifications, placebo years
- `referee.py`: district influence calculations and IV sensitivity arithmetic
- `verify.py`: coefficient and calculation checks
- `data/*.csv.gz`: derived district and newspaper tables
- `build_data.py`: rebuild from the original archived Stata data (not included in this lightweight download)

Sources: [Article](https://doi.org/10.1093/restud/rdac059); [replication archive](https://zenodo.org/records/6841670).
