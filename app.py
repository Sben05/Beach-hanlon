
"""Interactive sensitivity analysis for Beach & Hanlon (2023)."""

from html import escape
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import (
    CONTROL_NAMES,
    base_differences,
    estimate,
    load_data,
    placebo_years,
    specification_atlas,
)
from referee import (
    PUBLISHED_REDUCED_FORM,
    PUBLISHED_REDUCED_FORM_SE,
    direct_channel_sensitivity,
    influence_audit,
    influence_context,
)


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="How sensitive is the newspaper result?",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.set_option("client.showSidebarNavigation", False)

PAPER = "#f6f3ec"
INK = "#282521"
RED = "#8e483c"
BLUE = "#496276"


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    f"<style>{Path(__file__).with_name('style.css').read_text()}</style>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title):
    st.header(title)


def result(s):
    with st.container(border=True):
        st.markdown(s, unsafe_allow_html=True)


def note(s):
    st.caption(s)


def chart(fig, height=330, x=None, y=None):
    """
    Consistent research-paper chart style.

    Extra margins and automatic axis sizing prevent labels
    and long category names from being cut off.
    """

    fig.update_layout(
        template="none",
        height=max(height, 340),
        autosize=True,

        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",

        font=dict(
            family="Georgia, serif",
            size=13,
            color=INK,
        ),

        margin=dict(
            l=90,
            r=45,
            t=36,
            b=91,
            autoexpand=True,
        ),

        showlegend=False,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#fffdf8",
            bordercolor="#d6d8d0",
            font=dict(family="Georgia, serif", size=13, color=INK),
        ),
    )

    fig.update_xaxes(
        title=dict(text=x, standoff=20),
        automargin=True,

        showline=True,
        linecolor="#a69e91",

        tickfont=dict(size=12),
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        title=dict(text=y, standoff=16),
        automargin=True,
        ticklabeloverflow="allow",

        showline=False,
        gridcolor="#e5dfd4",

        tickfont=dict(size=12),
        zeroline=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={
            "displaylogo": False,
            "scrollZoom": False,
            "responsive": True,
        },
    )


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load():
    panel, annual, _, _ = load_data()
    return base_differences(panel), annual


@st.cache_data(show_spinner=False)
def specs(b):
    return specification_atlas(b)


@st.cache_data(show_spinner=False)
def leverage(b):
    return influence_audit(
        b,
        kmax=5,
        random_trials=300,
        seed=1877,
    )


@st.cache_data(show_spinner=False)
def time_scan(b, annual):
    return placebo_years(b, annual)


try:
    base, annual = load()

except (OSError, FileNotFoundError) as exc:
    st.error(
        f"Couldn't find the data files next to app.py: {exc}"
    )
    st.stop()


full = estimate(base, 3, 3, True)
atlas = specs(base)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown('<p class="paper-label">Beach &amp; Hanlon (2023)</p>', unsafe_allow_html=True)

st.title("How sensitive is their newspaper result?")

st.write(
    "One thing I kept wondering while reading the paper was "
    "how much the newspaper result depends on the choices "
    "behind the regression: the years, the controls, "
    "and the districts."
)
st.write("So I reproduced the estimate in Python and started changing a few things.")
st.write(
    "I made this for the presentation so we can try "
    "different assumptions together. If anyone has "
    "something they want to test, just suggest it."
)

st.markdown(
    """
    <ol class="event-timeline">
        <li><time>1875–1877</time><span>Earlier newspaper openings</span></li>
        <li><time>June 1877</time><span>Bradlaugh–Besant trial</span></li>
        <li><time>1878–1880</time><span>Later newspaper openings</span></li>
    </ol>
    """,
    unsafe_allow_html=True,
)

metrics = st.columns(3)
for column, label, value in zip(
    metrics,
    ("Paper's estimate", "My reproduction", "Districts in the sample"),
    (f'{PUBLISHED_REDUCED_FORM:+.3f}', f'{full["beta"]:+.3f}', str(full["n"])),
):
    with column:
        with st.container(border=True):
            st.metric(label, value)

note(
    "The outcome is the change in district birth rates. The coefficient "
    "is the difference in annual births per 1,000 women associated with "
    "one additional early-opening newspaper, relative to later-opening "
    "districts. It is a newspaper coefficient, not a direct estimate "
    "of the trial coverage effect."
)


# ============================================================
# 1. REGRESSION CHOICES
# ============================================================

section("1. What if we change the regression a little?")

st.write(
    "The regression compares changes in birth rates across districts "
    "with earlier and later newspaper openings. I changed the opening "
    "window, controls, and population weights to see how much the "
    "coefficient moves."
)

st.write(
    "In the chart, the window and weighting stay fixed while the "
    "control set changes. Choose a specification below."
)


# Store defaults once so button presets work correctly.

for key, val in {
    "opening_window": 3,
    "control_set": 3,
    "population_weighted": True,
    "omit_1877": False,
}.items():

    if key not in st.session_state:
        st.session_state[key] = val


def set_model(w, c, weighted, omit):
    st.session_state.update(
        opening_window=w,
        control_set=c,
        population_weighted=weighted,
        omit_1877=omit,
    )


def model_preset_active(w, c, weighted, omit):
    return (
        st.session_state["opening_window"] == w
        and st.session_state["control_set"] == c
        and st.session_state["population_weighted"] == weighted
        and st.session_state["omit_1877"] == omit
    )


# Presets and inputs stay together as one experiment panel.
with st.container(border=True):
    b1, b2, b3 = st.columns(3)

    with b1:
        st.button(
            "Original settings",
            use_container_width=True,
            type="primary" if model_preset_active(3, 3, True, False) else "secondary",
            on_click=set_model,
            args=(3, 3, True, False),
        )

    with b2:
        st.button(
            "No added controls",
            use_container_width=True,
            type="primary" if model_preset_active(3, 0, True, False) else "secondary",
            on_click=set_model,
            args=(3, 0, True, False),
        )

    with b3:
        st.button(
            "Leave out 1877 openings",
            use_container_width=True,
            type="primary" if model_preset_active(3, 3, True, True) else "secondary",
            on_click=set_model,
            args=(3, 3, True, True),
        )

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        window = st.selectbox(
            "How many years should we include?",
            [2, 3, 4, 5],
            key="opening_window",
        )
        weighted = st.checkbox(
            "Weight by district population",
            key="population_weighted",
        )

    with c2:
        controls = st.selectbox(
            "Which controls should we use?",
            list(range(4)),
            format_func=lambda j: CONTROL_NAMES[j],
            key="control_set",
        )
        omit = st.checkbox(
            "Leave out newspapers recorded as opening in 1877",
            key="omit_1877",
        )


now = estimate(
    base,
    window,
    controls,
    weighted,
    omit,
)

delta = now["beta"] - full["beta"]


# Regression comparison figure

fig = go.Figure()

SHORT_CONTROL_NAMES = [
    "Baseline",
    "+ district and marriage controls",
    "+ established newspapers",
    "+ lagged fertility",
]

for i in range(4):

    r = estimate(
        base,
        window,
        i,
        weighted,
        omit,
    )

    label = SHORT_CONTROL_NAMES[i]

    fig.add_trace(
        go.Scatter(
            x=[r["beta"]],
            y=[label],
            mode="markers",

            marker=dict(
                size=11 if i == controls else 8,
                color=RED if i == controls else BLUE,
                symbol=(
                    "square" if i == controls
                    else "circle"
                ),
            ),

            error_x=dict(
                type="data",
                array=[r["hi"] - r["beta"]],
                arrayminus=[r["beta"] - r["lo"]],
                width=3,
                thickness=1.2,
            ),

            hovertemplate=(
                f"{escape(CONTROL_NAMES[i])}<br>"
                "Estimate=%{x:.3f}<extra></extra>"
            ),
        )
    )


fig.add_vline(
    x=0,
    line_color="#969086",
    line_dash="dot",
)

fig.add_vline(
    x=full["beta"],
    line_color="#a6a09a",
    line_dash="dash",
)

fig.update_yaxes(autorange="reversed")

chart(
    fig,
    350,
    "Coefficient and 95% interval (HC1)",
)

note(
    "The red square is the selected control set; horizontal bars are "
    "95% HC1 intervals. The dotted vertical line marks zero and the "
    "dashed line marks the original estimate."
)


# Dynamic interpretation

result(
    f'This specification estimates <b>{now["beta"]:+.3f}</b> '
    'annual births per 1,000 women for one additional '
    'early-opening newspaper.'
)

st.write(
    f"That is **{delta:+.3f}** from the original estimate. "
    f"The regression uses **{now['n']}** districts; its 95% HC1 "
    f"interval is **[{now['lo']:+.2f}, {now['hi']:+.2f}]**."
)


if now["hi"] < 0:

    st.write(
        "The interval stays below zero. Under this specification, "
        "the estimate remains negative even after accounting for "
        "the uncertainty shown by this interval."
    )

elif now["lo"] > 0:

    st.write(
        "The interval lies above zero. This specification points "
        "in the opposite direction from the published estimate."
    )

else:

    st.write(
        "The interval includes zero. With these settings, the "
        "estimate is too imprecise to rule out a zero coefficient "
        "at the 5% level."
    )


# Results across all specifications

n_negative = int((atlas.beta < 0).sum())
n_sig = int((atlas.hi < 0).sum())

grid_summary = st.columns(3)
for column, label, value in zip(
    grid_summary,
    ("Specifications", "Negative estimates", "Intervals below zero"),
    (len(atlas), n_negative, n_sig),
):
    with column:
        with st.container(border=True):
            st.metric(label, value)

st.write(
    f"Across the grid, coefficients range from **{atlas.beta.min():+.2f}** "
    f"to **{atlas.beta.max():+.2f}**. The sign is stable across these "
    "choices, while the size and precision are less stable."
)

note(
    "These specifications reuse mostly the same districts; "
    "the counts are a sensitivity summary, not 32 independent tests. "
    "The paper's Appendix Table 25 already reports alternative "
    "opening windows and control sets."
)

st.markdown("**Uncertainty for the preferred specification**")
uncertainty = st.columns(2)
for column, label, value in zip(
    uncertainty,
    ("Published district-clustered SE", "Python first-difference HC1 SE"),
    (PUBLISHED_REDUCED_FORM_SE, full["se"]),
):
    with column:
        with st.container(border=True):
            st.metric(label, f"{value:.3f}")

with st.expander("A note on the calculations"):

    st.write(
        f"The published coefficient is {PUBLISHED_REDUCED_FORM:+.3f}; "
        f"my first-difference reproduction is {full['beta']:+.3f}. "
        "The point estimates match, while the standard errors use "
        "different calculations."
    )

    st.write(
        "Changing the opening window also changes which "
        "districts are included."
    )

    st.write(
        "The intervals in these Python re-estimations use HC1 "
        "standard errors on one first-difference observation per "
        "district. The paper estimates a two-period panel and clusters "
        "standard errors by district."
    )

    published_lo = PUBLISHED_REDUCED_FORM - 1.96 * PUBLISHED_REDUCED_FORM_SE
    published_hi = PUBLISHED_REDUCED_FORM + 1.96 * PUBLISHED_REDUCED_FORM_SE
    st.write(
        f"Using 1.96 times the published SE gives an approximate "
        f"95% interval of [{published_lo:+.2f}, {published_hi:+.2f}]. "
        f"The HC1 interval here is [{full['lo']:+.2f}, {full['hi']:+.2f}]."
    )

    st.write(
        "The point estimates reproduce Table 4."
    )


# ============================================================
# 2. DISTRICT INFLUENCE
# ============================================================

section("2. Are a few districts driving the result?")

st.write(
    "A few districts may carry a lot of weight in this regression. "
    "I checked how far the coefficient moves when districts are removed."
)

st.write(
    "The red line deliberately removes the district that most weakens "
    "the result at each step. The blue line shows the median from "
    "300 random removals of the same number of districts."
)

audit = leverage(base)
context = influence_context(audit, annual)

with st.container(border=True):
    k = st.slider(
        "How many districts should we remove?",
        min_value=0,
        max_value=5,
        value=5,
        step=1,
    )

path = audit["path"][k]


# Comparison figure

ks = np.arange(0, 6)

lo = [audit["baseline"]] + [
    audit["random"][i]["q05"]
    for i in range(1, 6)
]

hi = [audit["baseline"]] + [
    audit["random"][i]["q95"]
    for i in range(1, 6)
]

median = [audit["baseline"]] + [
    audit["random"][i]["median"]
    for i in range(1, 6)
]

greedy = [
    p["beta"] for p in audit["path"]
]


fig = go.Figure()

# 90% random-removal range

fig.add_trace(
    go.Scatter(
        x=np.r_[ks, ks[::-1]],
        y=np.r_[lo, hi[::-1]],
        fill="toself",
        fillcolor="rgba(73,98,118,.16)",
        line=dict(width=0),
        hoverinfo="skip",
    )
)

# Random-removal median

fig.add_trace(
    go.Scatter(
        x=ks,
        y=median,
        mode="lines+markers",

        line=dict(
            color=BLUE,
            width=2,
            dash="dot",
        ),

        marker=dict(size=6),
        name="Random median",
    )
)

# Targeted district removals

fig.add_trace(
    go.Scatter(
        x=ks,
        y=greedy,
        mode="lines+markers",

        line=dict(
            color=RED,
            width=3,
        ),

        marker=dict(size=7),
        name="Selected removals",
    )
)

# Selected number of districts

fig.add_trace(
    go.Scatter(
        x=[k],
        y=[path["beta"]],
        mode="markers",

        marker=dict(
            size=15,
            color=RED,
            line=dict(
                color=PAPER,
                width=2,
            ),
        ),
    )
)

fig.update_xaxes(dtick=1)

chart(
    fig,
    360,
    "Number of districts removed",
    "Estimated coefficient",
)


note("The shaded band covers the middle 90% of the random runs.")


if k:
    result(
        f'Removing the <b>{k}</b> {"district" if k == 1 else "districts"} '
        f'that most weaken the result moves the coefficient from '
        f'<b>{audit["baseline"]:+.3f}</b> to <b>{path["beta"]:+.3f}</b>.'
    )
else:
    result(f'With no districts removed, the coefficient is <b>{path["beta"]:+.3f}</b>.')


if k:

    rand = audit["random"][k]

    st.write(
        f"For {k} random removals, the median is **{rand['median']:+.2f}** "
        f"and the middle 90% runs from **{rand['q05']:+.2f}** "
        f"to **{rand['q95']:+.2f}**."
    )

else:

    st.write(
        f"At five targeted removals, the coefficient is "
        f"**{audit['path'][5]['beta']:+.2f}**. "
        f"The median from five random removals is "
        f"**{audit['random'][5]['median']:+.2f}**."
    )

    st.write(
        "Use the slider to see which districts "
        "the program selected."
    )


selected_five = context.loc[context.selection_order.notna()].sort_values(
    "selection_order"
)

st.markdown("**The five selected districts**")
district_table = selected_five[
    [
        "district", "population", "weight_share_pct", "1871_73",
        "1874_76", "pre_change", "wls_leverage", "beta_without", "delta",
    ]
].rename(columns={
    "district": "District",
    "population": "1871 population",
    "weight_share_pct": "Weight %",
    "1871_73": "Birth rate 1871–73",
    "1874_76": "Birth rate 1874–76",
    "pre_change": "Pre-trial change",
    "wls_leverage": "WLS leverage",
    "beta_without": "β without",
    "delta": "Δ from full",
})

st.dataframe(
    district_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "District": st.column_config.TextColumn(width="large"),
        "1871 population": st.column_config.NumberColumn(format="%d", width="small"),
        "Weight %": st.column_config.NumberColumn(format="%.2f", width="small"),
        "Birth rate 1871–73": st.column_config.NumberColumn(format="%.1f", width="small"),
        "Birth rate 1874–76": st.column_config.NumberColumn(format="%.1f", width="small"),
        "Pre-trial change": st.column_config.NumberColumn(format="%+.1f", width="small"),
        "WLS leverage": st.column_config.NumberColumn(format="%.3f", width="small"),
        "β without": st.column_config.NumberColumn(format="%+.3f", width="small"),
        "Δ from full": st.column_config.NumberColumn(format="%+.3f", width="small"),
    },
)

note(
    "Birth rates are annual births per 1,000 women. The two pre-trial "
    "columns are three-year averages; Δ from full removes each district "
    "alone, rather than following the five-step path."
)

with st.expander("How to read the district audit"):
    st.write(
        "Weight % is the district's share of 1871 population in the "
        "98-district regression sample. WLS leverage measures how unusual "
        "its weighted covariate row is in that regression; it combines "
        "population weight and covariate position."
    )
    st.write(
        f"Across the full sample, median weight share is "
        f"{context.weight_share_pct.median():.2f}%, median WLS leverage "
        f"is {context.wls_leverage.median():.3f}, and median pre-trial "
        f"birth-rate change is {context.pre_change.median():+.1f}."
    )
    remaining_base = base.loc[~base.district.isin(selected_five.district)]
    unweighted_full = estimate(base, 3, 3, False)
    unweighted_removed = estimate(remaining_base, 3, 3, False)
    st.write(
        f"Without population weights, the coefficient moves from "
        f"{unweighted_full['beta']:+.3f} to "
        f"{unweighted_removed['beta']:+.3f} when these same five "
        "districts are removed. The large change remains without "
        "population weights, so weighting alone does not explain it."
    )


# Inspect strategically removed districts

# Interpretation

st.write(
    "The targeted change is much larger than a typical random change. "
    "That shows the estimate is sensitive to particular districts; "
    "it does not show those districts are bad observations."
)

st.write(
    "My next question would be whether the influential "
    "districts share some historical feature, like unusual "
    "newspaper markets, fertility trends, or industrial "
    "composition. I haven't established that yet."
)

note(
    "These are deliberately chosen omissions, not justified "
    "sample exclusions. The random runs are a descriptive "
    "comparison, not a significance test."
)


with st.expander("Inspect individual districts"):
    if k:
        deleted = [p["district"] for p in audit["path"][1:k + 1]]
        lookup = {row["district"]: row for row in audit["rows"]}
        chosen = st.selectbox(
            "Look at one of the selected districts",
            deleted,
            key="district_detail",
        )
        row = lookup[chosen]
        st.write(
            f"**{chosen.title()}**: {row['region']} region; "
            f"population {row['population']:,} in 1871; "
            f"{row['early_openings']:.0f} earlier newspaper openings; "
            f"{row['articles']:.0f} recorded trial articles."
        )

    all_districts = sorted(audit["rows"], key=lambda row: row["district"])
    selected_district = st.selectbox(
        "Or try leaving out a particular district",
        ["None"] + [row["district"] for row in all_districts],
        key="class_district_choice",
    )
    if selected_district != "None":
        chosen_row = next(
            row for row in all_districts
            if row["district"] == selected_district
        )
        st.write(
            f"Without **{selected_district.title()}**, the coefficient "
            f"is **{chosen_row['beta_without']:+.3f}**, a change of "
            f"**{chosen_row['delta']:+.3f}** from the original. "
            f"That district has {chosen_row['articles']:.0f} recorded "
            "trial articles."
        )


# ============================================================
# 3. NEWSPAPER MECHANISM
# ============================================================

section(
    "3. What if newspapers affected fertility for other reasons?"
)

st.write(
    "The paper uses newspaper openings to predict trial coverage. "
    "Its interpretation depends on those openings affecting fertility "
    "through that coverage, rather than through other newspaper content."
)

st.write(
    "I can't measure that other effect here. The slider lets us "
    "assume a value and subtract it from the published coefficient."
)

# Initialize once: avoids Streamlit Session State warnings.

if "direct_effect" not in st.session_state:
    st.session_state["direct_effect"] = 0.0


def set_direct(g):
    st.session_state["direct_effect"] = g


with st.container(border=True):
    p1, p2, p3 = st.columns(3)

    with p1:
        st.button(
            "Assume zero",
            use_container_width=True,
            type="primary" if st.session_state["direct_effect"] == 0.0 else "secondary",
            on_click=set_direct,
            args=(0.0,),
        )

    with p2:
        st.button(
            "Try −1",
            use_container_width=True,
            type="primary" if st.session_state["direct_effect"] == -1.0 else "secondary",
            on_click=set_direct,
            args=(-1.0,),
        )

    with p3:
        st.button(
            "Try −2.675",
            use_container_width=True,
            type="primary" if st.session_state["direct_effect"] == -2.675 else "secondary",
            on_click=set_direct,
            args=(-2.675,),
        )

    st.divider()
    gamma = st.slider(
        "Assumed effect from other newspaper content",
        min_value=-3.5,
        max_value=1.5,
        step=0.025,
        key="direct_effect",
    )

direct = direct_channel_sensitivity(gamma)


# Sensitivity curve

grid = np.linspace(
    -3.5,
    1.5,
    201,
)

mid = PUBLISHED_REDUCED_FORM - grid
margin = 1.96 * PUBLISHED_REDUCED_FORM_SE

fig = go.Figure()

# Conditional 95% band

fig.add_trace(
    go.Scatter(
        x=np.r_[grid, grid[::-1]],
        y=np.r_[
            mid - margin,
            (mid + margin)[::-1],
        ],

        fill="toself",
        line=dict(width=0),
        fillcolor="rgba(73,98,118,.14)",
        hoverinfo="skip",
    )
)

# Adjusted effect

fig.add_trace(
    go.Scatter(
        x=grid,
        y=mid,
        mode="lines",
        line=dict(
            color=BLUE,
            width=2,
        ),
    )
)

# Audience-selected assumption

fig.add_trace(
    go.Scatter(
        x=[gamma],
        y=[direct["adjusted_per_opening"]],
        mode="markers",

        marker=dict(
            size=12,
            color=RED,
        ),
    )
)

fig.add_hline(
    y=0,
    line_color="#8e887f",
    line_dash="dash",
)

fig.add_vline(
    x=direct["gamma_ci_boundary"],
    line_color="#b78d80",
    line_dash="dot",
)

chart(
    fig,
    360,
    "Assumed effect from other newspaper content",
    "Remaining effect per opening",
)


# Dynamic interpretation

result(
    f'With an assumed other-content effect of <b>{gamma:+.3f}</b> '
    f'per opening, the remaining estimate is '
    f'<b>{direct["adjusted_per_opening"]:+.3f}</b>.'
)

if direct["ci_lo"] <= 0 <= direct["ci_hi"]:

    st.write(
        "With this assumption, the adjusted interval "
        "includes zero. The data would no longer rule "
        "out a zero remaining effect at the 5% level, "
        "conditional on the assumed other channel."
    )

elif direct["ci_hi"] < 0:

    st.write(
        "With this assumption, the remaining effect "
        "is still negative, and its conditional interval "
        "stays below zero."
    )

else:

    st.write(
        "At this assumption, the remaining effect "
        "and its conditional interval are positive."
    )


st.write(
    f"An assumed other-content effect of about "
    f"**{direct['gamma_ci_boundary']:+.2f} per opening** makes the "
    "conditional interval touch zero. That is a threshold from the "
    "calculation, not an estimate of what other content did."
)

note(
    "We're choosing the additional effect ourselves, "
    "not estimating it from the data. This is a sensitivity "
    "calculation, not a test of whether the alternative "
    "effect existed."
)


with st.expander("How this calculation works"):

    st.write(
        f"The conditional 95% interval for the remaining effect per "
        f"opening is [{direct['ci_lo']:+.2f}, {direct['ci_hi']:+.2f}]."
    )

    st.write(
        "Using the published first stage of 9.756 extra recorded "
        "trial articles per opening, this implies "
        f"{direct['adjusted_per_article']:+.3f} births per 1,000 "
        "women per predicted article."
    )

    st.write(
        "The calculation subtracts the assumed other-news "
        f"effect from the published {PUBLISHED_REDUCED_FORM:+.3f} "
        "estimate per opening."
    )

    st.write(
        f"It uses the reported standard error of "
        f"{PUBLISHED_REDUCED_FORM_SE:.3f} "
        "and holds the assumption fixed."
    )

    st.write(
        "The implied effect per trial article also uses "
        "the published first stage. This is not a formal "
        "IV confidence set, and uncertainty in the "
        "first stage is not added here."
    )


# ============================================================
# 4. TIMING
# ============================================================

section("4. What happens if we move the year?")

st.write(
    "I moved the comparison year and reran the regression. "
    "If the trial timing matters, 1877 should look different "
    "from nearby years."
)

scan = time_scan(
    base,
    annual,
).sort_values("focal")

years = scan.focal.astype(int).tolist()

with st.container(border=True):
    year = st.select_slider(
        "Which year should we try?",
        options=years,
        value=1877,
    )

r = scan.loc[
    scan.focal == year
].iloc[0]


# Timing comparison figure

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=scan.focal,
        y=scan.beta,
        mode="lines+markers",

        line=dict(
            color=BLUE,
            width=2,
        ),

        marker=dict(
            size=7,
            color=BLUE,
        ),

        error_y=dict(
            type="data",
            array=scan.hi - scan.beta,
            arrayminus=scan.beta - scan.lo,
            width=2,
            thickness=1,
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=[year],
        y=[r.beta],
        mode="markers",

        marker=dict(
            size=15,
            color=RED,
            line=dict(
                color=PAPER,
                width=2,
            ),
        ),
    )
)

fig.add_vline(
    x=1877,
    line_color="#aaa196",
    line_dash="dot",
)

fig.add_hline(
    y=0,
    line_color="#aaa196",
    line_dash="dash",
)

fig.update_xaxes(dtick=1)

chart(
    fig,
    360,
    "Assumed break year",
    "Estimated coefficient (95% HC1 interval)",
)


result(
    f'Using <b>{year}</b> as the break year, '
    f'the estimate is <b>{r.beta:+.3f}</b> '
    f'({int(r.n)} districts).'
)

st.write(
    f"Its 95% HC1 interval is **[{r.lo:+.2f}, {r.hi:+.2f}]**."
)


strongest_year = int(
    scan.loc[
        scan.beta.idxmin(),
        "focal",
    ]
)

reference = scan.loc[
    scan.focal == 1877,
    "beta",
].iloc[0]


if year == strongest_year:

    st.write(
        f"**{year} has the most negative point estimate** "
        f"among the {len(scan)} break years I tried."
    )

else:

    st.write(
        f"The 1877 estimate is **{reference:+.3f}**, "
        f"the most negative of the {len(scan)} dates tried."
    )


other_years = scan.loc[
    scan.focal != 1877
]

other_exclude_zero = int(
    (
        (other_years.hi < 0)
        |
        (other_years.lo > 0)
    ).sum()
)

st.write(
    f"**{len(other_years) - other_exclude_zero} "
    f"of the {len(other_years)} other dates** "
    "have intervals including zero."
)

st.write(
    "The pattern is consistent with the paper's timing argument, "
    "but the scan alone cannot identify the trial's effect."
)

note(
    "The paper also varies focal years in Figure 6. This related "
    "check changes periods and district samples across years, so "
    "it is not separate proof of causation."
)


# ============================================================
# CONCLUSION
# ============================================================

section("So, what did I take away from this?")

st.write(
    "The checks answer different questions. Taken together, they "
    "show where the newspaper result is stable and where it remains open."
)

conclusion_top = st.columns(2)
with conclusion_top[0]:
    with st.container(border=True):
        st.markdown("**Regression choices**")
        st.write(
            f"All {n_negative} of {len(atlas)} specifications have negative "
            f"coefficients; {n_sig} have 95% HC1 intervals below zero. "
            "The direction holds across this grid, but precision varies."
        )

with conclusion_top[1]:
    with st.container(border=True):
        st.markdown("**District influence**")
        st.write(
            f"Deliberately removing five influential districts changes the "
            f"coefficient from {audit['baseline']:+.2f} to "
            f"{audit['path'][5]['beta']:+.2f}. That is a stress test, "
            "not a reason to exclude them."
        )

conclusion_bottom = st.columns(2)
with conclusion_bottom[0]:
    with st.container(border=True):
        st.markdown("**Other newspaper content**")
        st.write(
            f"An unmeasured effect of about "
            f"{direct['gamma_ci_boundary']:+.2f} per opening would make "
            "the conditional interval touch zero. The app does not "
            "estimate whether that effect existed."
        )

with conclusion_bottom[1]:
    with st.container(border=True):
        st.markdown("**Timing**")
        st.write(
            f"1877 has the most negative coefficient among the "
            f"{len(scan)} break years tried. The periods and district "
            "samples change, so this is a timing check rather than "
            "separate proof of causation."
        )

st.write(
    "I would look next at the influential districts, their earlier "
    "fertility patterns, and what their newspapers published. "
    "That evidence could help judge the trial-coverage explanation."
)


# ============================================================
# DATA AND SOURCES
# ============================================================

st.divider()

with st.expander("Data and sources"):

    st.write(
        "Beach & Hanlon (2023), Review of Economic Studies, "
        "Tables 4–5."
    )

    st.write(
        "The district data come from the authors' "
        "public replication archive."
    )

    st.write(
        "The Python regressions use first differences, "
        "regional indicators, and the same control families "
        "as Table 4."
    )

    st.write(
        "The district-removal search and the other-news "
        "sensitivity calculation are additional analyses."
    )

    st.markdown(
        "[Published paper]"
        "(https://doi.org/10.1093/restud/rdac059)"
        " | "
        "[Author PDF]"
        "(https://walkerhanlon.com/papers/beach_hanlon_demographic_transition.pdf)"
        " | "
        "[Replication archive]"
        "(https://zenodo.org/records/6841670)"
    )

    st.download_button(
        "Download the 32 specification results",
        atlas.to_csv(index=False),
        file_name="1877_specifications.csv",
        mime="text/csv",
    )

st.write("Thanks for the time and attention during the presentation!")
