"""Streamlit article explaining the Black-Litterman vs. Mean-Variance backtest.

Presentation layer only. Every number on the page is read from the committed
artefacts in `results/`, or computed directly from them. Nothing here touches
the backtest, the models, or the views.

Run:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

STRATEGIES = [
    ("equal_weight",    "Equal Weight"),
    ("spy",             "S&P 500 (SPY)"),
    ("mvo",             "Mean-Variance"),
    ("black_litterman", "Black-Litterman"),
]

# Plain-language names for the ten sector ETFs. The reader never sees a ticker.
SECTOR_NAMES = {
    "XLK":  "Technology",
    "XLF":  "Banks and finance",
    "XLV":  "Health care",
    "XLY":  "Consumer brands",
    "XLI":  "Industrials",
    "XLP":  "Food and household",
    "XLE":  "Energy",
    "XLU":  "Utilities",
    "XLRE": "Real estate",
    "XLB":  "Materials",
}

# The five sectors that Mean-Variance ever held as its largest position get a
# colour. The other five stay grey, so the chart has five labels, not ten.
SECTOR_COLOURS = {
    "XLK": "#4C8DFF",
    "XLE": "#FF7A45",
    "XLV": "#35C88F",
    "XLF": "#C084FC",
    "XLI": "#FFD166",
}
MUTED = "#4A5364"

INK = "#ECEFF4"
DIM = "#98A2B3"
GRID = "#242B38"
BG = "#0E1117"

BLUE = "#4C8DFF"
AMBER = "#FF9F45"

st.set_page_config(
    page_title="What happens when a portfolio chases the winner",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# chart template
# ---------------------------------------------------------------------------

pio.templates["article"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                  size=16, color=INK),
        title=dict(font=dict(size=26, color=INK), x=0, xanchor="left",
                   y=0.94, yanchor="top"),
        margin=dict(l=70, r=215, t=125, b=60),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=GRID,
                   ticks="outside", tickcolor=GRID,
                   tickfont=dict(size=15, color=DIM)),
        yaxis=dict(gridcolor=GRID, zeroline=False, showline=False,
                   tickfont=dict(size=15, color=DIM)),
        hovermode="x unified",
        showlegend=False,
    )
)
pio.templates.default = "article"

CHART_CONFIG = {"displayModeBar": False}


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "summary.csv", index_col="strategy")


@st.cache_data
def load_returns(key: str) -> pd.Series:
    df = pd.read_csv(RESULTS / f"returns_{key}.csv", index_col=0, parse_dates=True)
    return df["return"]


@st.cache_data
def load_weights(key: str) -> pd.DataFrame:
    w = pd.read_csv(RESULTS / f"weights_{key}.csv", index_col=0, parse_dates=True)
    # The optimiser returns values like 4e-17 for an empty slot. Clip so the
    # stacked areas do not carry invisible negative slivers.
    return w.clip(lower=0.0)


@st.cache_data
def equity_curves() -> pd.DataFrame:
    """Wealth after each month's return, dated by the month it was earned.

    The committed return series is indexed by the decision date t, and the
    value stored there is the return earned over t to t+1 month (see
    backtest/engine.py, `_next_month_return`). Shifting the index forward one
    month-end puts each point on the calendar month it actually describes, so
    chart callouts name the right month.
    """
    eq = pd.DataFrame({
        name: (1.0 + load_returns(key)).cumprod()
        for key, name in STRATEGIES
    })
    eq.index = eq.index + pd.offsets.MonthEnd(1)
    return eq


summary = load_summary()
bl = summary.loc["Black-Litterman"]
mv = summary.loc["Mean-Variance"]

w_mvo = load_weights("mvo")
w_bl = load_weights("black_litterman")

# Derived facts, computed from the committed weight files at load time.
mvo_top_share = w_mvo.max(axis=1)
bl_top_share = w_bl.max(axis=1)
mvo_months_over_90 = int((mvo_top_share > 0.90).sum())
n_months = len(w_mvo)
first_date = w_mvo.index.min()
last_date = w_mvo.index.max()


# ---------------------------------------------------------------------------
# small layout helpers
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2.2rem;}
      h1.lede {font-size: 3.1rem; line-height: 1.12; font-weight: 700;
               letter-spacing: -0.02em; margin-bottom: 0.7rem;}
      p.method {font-size: 1.12rem; color: #98A2B3; line-height: 1.6;
                max-width: 46rem; margin-bottom: 1.4rem;}
      div.warnbox {border-left: 4px solid #FF9F45; background:#161B25;
                   padding: 0.9rem 1.2rem; border-radius: 4px;
                   font-size: 1.05rem; color:#D7DCE5; max-width: 46rem;
                   margin-bottom: 2.2rem;}
      p.body {font-size: 1.22rem; line-height: 1.65; max-width: 44rem;
              color:#DDE2EA;}
      h2.sec {font-size: 2.0rem; font-weight: 700; letter-spacing:-0.01em;
              margin-top: 2.6rem; margin-bottom: 0.4rem;}
      div.card {background:#161B25; border-radius:8px; padding:1.3rem 1.4rem;
                height:100%;}
      div.card .lbl {font-size:1.0rem; color:#98A2B3; line-height:1.35;}
      div.card .plain {font-size:0.98rem; color:#7D8698; line-height:1.45;
                       margin-top:0.55rem;}
      div.card .row {display:flex; justify-content:space-between;
                     align-items:baseline; margin-top:0.85rem;}
      div.card .nm {font-size:1.0rem;}
      div.card .vl {font-size:2.5rem; font-weight:700; letter-spacing:-0.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)


def compare_card(label: str, plain: str, mvo_val: str, bl_val: str,
                 bl_is_better: bool) -> str:
    good, bad = "#35C88F", "#FF7A45"
    bl_col = good if bl_is_better else bad
    mv_col = bad if bl_is_better else good
    return (
        f"<div class='card'><div class='lbl'>{label}</div>"
        f"<div class='plain'>{plain}</div>"
        f"<div class='row'><span class='nm' style='color:{mv_col}'>Chases the winner</span>"
        f"<span class='vl' style='color:{mv_col}'>{mvo_val}</span></div>"
        f"<div class='row'><span class='nm' style='color:{bl_col}'>Stays spread out</span>"
        f"<span class='vl' style='color:{bl_col}'>{bl_val}</span></div></div>"
    )


# ---------------------------------------------------------------------------
# the lede
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 class='lede'>One way of building a portfolio put every dollar into "
    "a single slice of the stock market. It made less money and lost far more "
    "along the way.</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<p class='method'>Two standard methods for splitting money across the ten US stock "
    f"market sectors, tested against real prices with one decision each month. "
    f"Mean-variance picks whatever looks best from the recent past. "
    f"Black-Litterman starts from how the whole market is already invested and adjusts "
    f"from there. "
    f"{n_months} monthly decisions, the first at the end of "
    f"{first_date.strftime('%B %Y')} and the last at the end of "
    f"{last_date.strftime('%B %Y')}, with returns running through December 2025."
    f"</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='warnbox'><b>Read this first.</b> Nothing here was traded. "
    "These are the results of replaying two rules against historical market "
    "prices to see what they would have done. Real money behaves differently, "
    "past prices do not repeat, and this is not investment advice.</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# the centrepiece: what each method actually held
# ---------------------------------------------------------------------------

st.markdown("<h2 class='sec'>What each method actually held</h2>",
            unsafe_allow_html=True)

st.markdown(
    f"<p class='body'>Every month the rule looks at the past three years of "
    f"prices and decides how to split the money across ten sectors. Each "
    f"coloured band below shows one sector, with the height showing how much "
    f"of the portfolio is in it. If one colour fills the whole height, "
    f"everything is in that one sector.</p>",
    unsafe_allow_html=True,
)


key_bits = "".join(
    f"<span style='display:inline-block;margin-right:1.5rem;white-space:nowrap'>"
    f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;"
    f"background:{SECTOR_COLOURS[t]};vertical-align:-2px;margin-right:0.45rem'></span>"
    f"{SECTOR_NAMES[t]}</span>"
    for t in ["XLK", "XLE", "XLV", "XLF", "XLI"]
)
st.markdown(
    f"<p style='font-size:1.02rem;color:#B6BECC;max-width:52rem;"
    f"margin-bottom:2.1rem'>{key_bits}"
    f"<span style='display:inline-block;white-space:nowrap'>"
    f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;"
    f"background:{MUTED};vertical-align:-2px;margin-right:0.45rem'></span>"
    f"the other five sectors</span></p>",
    unsafe_allow_html=True,
)


def weights_figure(w: pd.DataFrame, title: str, subtitle: str,
                   annotations: list[dict], band_labels: list[dict],
                   height: int = 470) -> go.Figure:
    fig = go.Figure()
    # Draw the grey sectors first so the coloured story sits on top.
    order = [t for t in w.columns if t not in SECTOR_COLOURS] + \
            [t for t in w.columns if t in SECTOR_COLOURS]
    for ticker in order:
        coloured = ticker in SECTOR_COLOURS
        fig.add_trace(go.Scatter(
            x=w.index, y=w[ticker] * 100,
            name=SECTOR_NAMES[ticker],
            mode="lines", line=dict(width=0),
            stackgroup="one",
            fillcolor=SECTOR_COLOURS.get(ticker, MUTED),
            hovertemplate="%{y:.0f}% " + SECTOR_NAMES[ticker] + "<extra></extra>",
        ))

    # Labels sit inside the bands themselves, so there is no legend to decode.
    for lab in band_labels:
        fig.add_annotation(showarrow=False, xanchor="center", yanchor="middle",
                           font=dict(size=17, color="#0E1117"), **lab)

    for a in annotations:
        fig.add_annotation(**a)

    fig.update_layout(
        title=dict(text=f"{title}<br><span style='font-size:16px;color:{DIM}'>"
                        f"{subtitle}</span>"),
        height=height,
        margin=dict(l=70, r=40, t=125, b=60),
        yaxis=dict(range=[0, 100],
                   tickmode="array", tickvals=[0, 25, 50, 75, 100],
                   ticktext=["0", "25%", "50%", "75%", "100%"]),
        xaxis=dict(range=[w.index.min(), w.index.max()]),
    )
    return fig


arrow = dict(showarrow=True, arrowhead=0, arrowsize=1, arrowwidth=1.6,
             arrowcolor=INK, align="left", bgcolor="rgba(14,17,23,0.85)",
             bordercolor=GRID, borderwidth=1, borderpad=6,
             font=dict(size=15, color=INK))

st.plotly_chart(
    weights_figure(
        w_mvo,
        "Mean-variance held one sector and nothing else, for years at a time",
        f"Share of the portfolio in each sector. It sat above 90 percent in a "
        f"single sector in {mvo_months_over_90} of {n_months} months.",
        [
            dict(x="2020-06-30", y=52,
                 text="Technology was the biggest holding<br>"
                      "for 47 months straight, and the only<br>"
                      "holding for 12 of them",
                 ax=0, ay=0, **arrow),
            dict(x="2023-09-30", y=45,
                 text="Switched to energy in January 2023,<br>"
                      "after technology had already fallen",
                 ax=-10, ay=-115, **arrow),
        ],
        band_labels=[
            dict(x="2020-03-31", y=88, text="TECHNOLOGY"),
            dict(x="2024-03-31", y=20, text="ENERGY"),
        ],
    ),
    use_container_width=True, config=CHART_CONFIG, theme=None,
)

st.plotly_chart(
    weights_figure(
        w_bl,
        "Black-Litterman never put more than half in any one sector",
        f"Same ten sectors, same month-by-month decisions. Its largest single "
        f"holding is typically {bl_top_share.median() * 100:.0f} percent.",
        [
            dict(x="2020-09-30", y=50,
                 text="Eight sectors held in a typical month",
                 ax=0, ay=0, **arrow),
        ],
        band_labels=[
            dict(x="2021-03-31", y=63, text="TECHNOLOGY"),
            dict(x="2024-06-30", y=93, text="ENERGY"),
            dict(x="2022-09-30", y=82, text="HEALTH CARE"),
            dict(x="2024-09-30", y=71, text="BANKS"),
            dict(x="2021-06-30", y=16, text="THE OTHER FIVE SECTORS"),
        ],
    ),
    use_container_width=True, config=CHART_CONFIG, theme=None,
)

st.markdown(
    "<p class='body'>That is the whole argument in two pictures. The first "
    "rule trusts the average return of the recent past, and the sector with "
    "the highest recent average wins the entire portfolio. When the leader "
    "changes, the portfolio moves in one lurch. The second rule starts from "
    "the way the market as a whole is already invested and only tilts away "
    "from it, so it stays mixed.</p>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# what it cost
# ---------------------------------------------------------------------------

st.markdown("<h2 class='sec'>What the concentration cost</h2>",
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="medium")
c1.markdown(compare_card(
    "Worst fall from a high point",
    "The deepest drop from peak to bottom if you held through without selling.",
    f"{mv['max_drawdown'] * 100:.1f}%",
    f"{bl['max_drawdown'] * 100:.1f}%",
    bl_is_better=True,
), unsafe_allow_html=True)
c2.markdown(compare_card(
    "Return per unit of risk",
    "How much return each rule earned relative to how much it bounced up and down. "
    "Higher is better.",
    f"{mv['sharpe']:.2f}",
    f"{bl['sharpe']:.2f}",
    bl_is_better=True,
), unsafe_allow_html=True)
c3.markdown(compare_card(
    "Average return per year",
    "Annual growth across the full test, before taxes and trading costs.",
    f"{mv['annual_return'] * 100:.1f}%",
    f"{bl['annual_return'] * 100:.1f}%",
    bl_is_better=True,
), unsafe_allow_html=True)

st.markdown(
    f"<p class='body' style='margin-top:1.6rem'>The concentrated rule took "
    f"bigger swings and did not get paid for them. It earned "
    f"{mv['annual_return'] * 100:.1f} percent a year against "
    f"{bl['annual_return'] * 100:.1f} percent, and to get it you had to hold "
    f"on through a fall of {abs(mv['max_drawdown']) * 100:.1f} percent instead "
    f"of {abs(bl['max_drawdown']) * 100:.1f} percent.</p>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# the money chart
# ---------------------------------------------------------------------------

eq = equity_curves()
dd = eq / eq.cummax() - 1.0

fig = go.Figure()
for name, colour, width in [
    ("Equal Weight", MUTED, 2),
    ("S&P 500 (SPY)", MUTED, 2),
    ("Mean-Variance", AMBER, 3.4),
    ("Black-Litterman", BLUE, 3.4),
]:
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq[name], mode="lines", name=name,
        line=dict(color=colour, width=width),
        hovertemplate="$%{y:.2f} " + name + "<extra></extra>",
    ))
ends = [
    ("Mean-Variance", AMBER, "Chases the winner"),
    ("Black-Litterman", BLUE, "Stays spread out"),
    ("S&P 500 (SPY)", DIM, "Whole US market"),
    ("Equal Weight", DIM, "Equal split"),
]
# Push apart any two labels that would otherwise sit on top of each other.
ends.sort(key=lambda e: eq[e[0]].iloc[-1], reverse=True)
placed: list[float] = []
gap = (eq.max().max() - 1.0) * 0.085
for name, colour, label in ends:
    y = float(eq[name].iloc[-1])
    while placed and abs(y - placed[-1]) < gap:
        y = placed[-1] - gap
    placed.append(y)
    fig.add_annotation(
        x=1.015, y=y, xref="paper", yref="y",
        text=f"{label}  <b>${eq[name].iloc[-1]:.2f}</b>",
        showarrow=False, xanchor="left",
        font=dict(size=15, color=colour),
    )
fig.add_hline(y=1.0, line=dict(color=GRID, width=1))
fig.add_annotation(
    x="2022-09-30", y=1.35,
    text="Through 2022 the concentrated rule<br>was still mostly in technology",
    ax=0, ay=-70, **arrow,
)
fig.update_layout(
    title=dict(text="Both ended in roughly the same place, by very different "
                    "roads<br><span style='font-size:16px;color:" + DIM +
                    "'>What one dollar became from the end of November 2018, "
                    "the first month the two rules were graded on.</span>"),
    height=520,
    yaxis=dict(tickprefix="$", tickformat=".1f"),
)
st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, theme=None)

fig = go.Figure()
for name, colour, width in [
    ("Mean-Variance", AMBER, 3.0),
    ("Black-Litterman", BLUE, 3.0),
]:
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd[name] * 100, mode="lines", name=name,
        line=dict(color=colour, width=width),
        fill="tozeroy",
        fillcolor="rgba(255,159,69,0.15)" if name == "Mean-Variance"
        else "rgba(76,141,255,0.15)",
        hovertemplate="%{y:.1f}% " + name + "<extra></extra>",
    ))
fig.add_hline(y=-20, line=dict(color=GRID, width=1, dash="dot"))
fig.add_annotation(x=dd.index.min(), y=-20, text="a 20 percent fall",
                   showarrow=False, xanchor="left", yanchor="bottom", yshift=4,
                   font=dict(size=14, color=DIM))
fig.add_annotation(
    x="2022-09-30", y=-33.1,
    text="Worst point, end of September 2022:<br>down 33.1 percent",
    ax=60, ay=45, **arrow,
)
fig.add_annotation(x=1.015, y=-2, xref="paper", yref="y",
                   text="Stays spread out", showarrow=False, xanchor="left",
                   font=dict(size=15, color=BLUE))
fig.add_annotation(x=1.015, y=-9, xref="paper", yref="y",
                   text="Chases the winner", showarrow=False, xanchor="left",
                   font=dict(size=15, color=AMBER))
fig.update_layout(
    title=dict(text="How far underwater you were at any moment<br>"
                    "<span style='font-size:16px;color:" + DIM +
                    "'>Distance below the highest value the portfolio had "
                    "reached so far. Zero means a new high.</span>"),
    height=440,
    yaxis=dict(ticksuffix="%"),
)
st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG, theme=None)


# ---------------------------------------------------------------------------
# the caveats and the numbers
# ---------------------------------------------------------------------------

st.markdown("<h2 class='sec'>How the test was run</h2>", unsafe_allow_html=True)
st.markdown(
    f"<p class='body'>At the end of each month the rule is shown the previous "
    f"three years of daily prices and nothing after that date. It picks its "
    f"split, and then the next month happens. The score is whatever that split "
    f"actually earned. Neither rule is ever allowed to see the month it is "
    f"being graded on. That constraint is the point: any rule can be made to "
    f"look good on data it has already seen.</p>"
    f"<p class='body'>Daily prices run from October 2015, three years before "
    f"the first decision, because the first decision needs a full three years "
    f"of history behind it. The ten sectors are the SPDR sector funds, and "
    f"the comparison lines are an equal split across all ten and the S&amp;P 500 "
    f"itself. Neither rule is allowed to borrow or bet against a sector.</p>",
    unsafe_allow_html=True,
)

with st.expander("All the numbers"):
    display = summary.copy()
    for c in ["annual_return", "annual_vol", "max_drawdown"]:
        display[c] = display[c].map(
            lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "n/a")
    for c in ["sharpe", "sortino", "turnover", "concentration_hhi"]:
        display[c] = display[c].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "n/a")
    display = display.rename(columns={
        "annual_return":     "Return a year",
        "annual_vol":        "Size of swings",
        "sharpe":            "Return per unit of risk (Sharpe)",
        "sortino":           "Same, counting only downside (Sortino)",
        "max_drawdown":      "Worst fall from a high",
        "turnover":          "Share of portfolio traded a month",
        "concentration_hhi": "How concentrated (0 spread, 1 all in one)",
    })
    st.dataframe(display, use_container_width=True)
    st.markdown(
        f"<p style='color:{DIM};font-size:0.98rem'>The concentration column is "
        f"the Herfindahl index, the sum of the squared weights. A portfolio "
        f"split evenly across ten sectors scores 0.10. A portfolio entirely in "
        f"one sector scores 1.00. Mean-variance scores "
        f"{mv['concentration_hhi']:.2f} and Black-Litterman "
        f"{bl['concentration_hhi']:.2f}. The S&amp;P 500 row has no weights or "
        f"trading of its own, so those two columns are blank.</p>",
        unsafe_allow_html=True,
    )

with st.expander("The math behind Black-Litterman"):
    st.markdown("**Ledoit-Wolf shrinkage covariance.**")
    st.latex(r"\hat{\Sigma} = \delta F + (1 - \delta) S")
    st.markdown("**Implied equilibrium returns.** Reverse-optimise the "
                "consensus portfolio:")
    st.latex(r"\Pi = \lambda \, \Sigma \, w_{mkt}")
    st.markdown("**Posterior.** The Black-Litterman master formula is the "
                "Gaussian product of prior and view:")
    st.latex(
        r"\hat{\mu} = \left[(\tau \Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}"
        r"\left[(\tau \Sigma)^{-1} \Pi + P^\top \Omega^{-1} Q\right]"
    )
    st.latex(
        r"\hat{\Sigma} = \Sigma + \left[(\tau \Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}"
    )

st.divider()
st.markdown(
    f"<p style='color:{DIM};font-size:0.98rem'>A backtest, not a track record. "
    f"No money was invested and this is not investment advice. Source code and "
    f"data at github.com/patrickt6/stable-sharpe. Patrick Taylor, Queen's "
    f"University.</p>",
    unsafe_allow_html=True,
)
