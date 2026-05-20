"""
SDG 3 – Child Nutrition Dashboard
Analytics Techniques and Tools — Finals ALA
WVSU Information Systems | Business Analytics

Run: python3 -m streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pycountry
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SDG 3 · Child Nutrition Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .dashboard-title { font-size: 2rem; font-weight: 700; color: #1E3A5F; margin-bottom: 0; }
    .dashboard-sub  { font-size: 1rem; color: #6B7280; margin-top: 4px; }
    .section-header { font-size: 1.15rem; font-weight: 600; color: #1E3A5F;
                      border-left: 4px solid #2563EB; padding-left: 10px; margin: 1rem 0 0.5rem; }
    .insight-box { background: #EFF6FF; border-left: 4px solid #2563EB;
                   padding: 10px 14px; border-radius: 6px; margin: 6px 0;
                   font-size: 0.9rem; color: #1E3A5F; }
    .footer-note { font-size: 0.78rem; color: #9CA3AF; text-align: center; padding-top: 16px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("unicef_child_nutrition_merged.csv")
    df.loc[df["country"].isin(["USA", "CAN"]), "un_region"] = "NA"
    df["un_region"] = df["un_region"].fillna("Unknown")
    df["un_region"] = df["un_region"].replace({"nan": "Unknown", "None": "Unknown"})
    df = df[~df["income_group"].isin(["Not Classified", "nan", "None"])]
    df = df.dropna(subset=["thinness_pct", "obesity_pct", "overweight_pct"])
    df["income_group"] = df["income_group"].str.replace(" Income", "").str.strip()

    iso3_names = {c.alpha_3: c.name for c in pycountry.countries if hasattr(c, "alpha_3")}
    df["country_name"] = df["country"].map(iso3_names).fillna(df["country"])
    return df

@st.cache_data
def run_regression(df):
    formula = (
        "thinness_pct ~ obesity_pct + year + "
        "C(income_group, Treatment('High')) + "
        "C(un_region, Treatment('WE'))"
    )
    ols = smf.ols(formula=formula, data=df).fit()
    robust = ols.get_robustcov_results(cov_type="HC3")
    return ols, robust

df = load_data()
ols_model, robust_model = run_regression(df)

params_s = pd.Series(robust_model.params, index=ols_model.params.index)
pvals_s  = pd.Series(robust_model.pvalues, index=ols_model.params.index)
ci_df    = pd.DataFrame(robust_model.conf_int(), index=ols_model.params.index)

ind_labels = {"thinness_pct": "Thinness (%)", "obesity_pct": "Obesity (%)", "overweight_pct": "Overweight (%)"}
ind_colors = {"thinness_pct": "YlOrRd", "obesity_pct": "RdPu", "overweight_pct": "YlOrBr"}

with st.sidebar:
    st.markdown("### 🌍 UNICEF")
    st.markdown("## Filters")
    year_range = st.slider("Year Range", int(df["year"].min()), int(df["year"].max()), (2000, 2022))
    selected_regions = st.multiselect(
        "UN Region(s)",
        options=sorted(df["un_region"].fillna("Unknown").unique()),
        default=sorted(df["un_region"].fillna("Unknown").unique())
    )
    selected_income = st.multiselect("Income Group(s)", options=["Low", "Lower Middle", "Upper Middle", "High"], default=["Low", "Lower Middle", "Upper Middle", "High"])
    indicator = st.selectbox("Primary Indicator", options=["thinness_pct", "obesity_pct", "overweight_pct"],
        format_func=lambda x: {"thinness_pct": "🔵 Thinness (%)", "obesity_pct": "🔴 Obesity (%)", "overweight_pct": "🟠 Overweight (%)"}[x])
    st.markdown("---")
    st.markdown("**Data Source**")
    st.caption("UNICEF Global Database\nChild Nutrition (SACA), Aug 2025\nAge group: 5–19 years")
    st.markdown("**Course**")
    st.caption("Analytics Techniques and Tools — Finals ALA\nWest Visayas State University")

mask = (
    (df["year"] >= year_range[0]) & (df["year"] <= year_range[1]) &
    (df["un_region"].isin(selected_regions)) & (df["income_group"].isin(selected_income))
)
dff = df[mask].copy()

_, col_title = st.columns([1, 9])
with col_title:
    st.markdown('<p class="dashboard-title">🌍 SDG 3 — Child Malnutrition Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="dashboard-sub">Understanding drivers of thinness, obesity & overweight among children 5–19 years'
        f' &nbsp;|&nbsp; {year_range[0]}–{year_range[1]} &nbsp;|&nbsp; {dff["country"].nunique()} countries</p>',
        unsafe_allow_html=True)

st.markdown("---")

def delta_kpi(col):
    early = dff[dff["year"] == dff["year"].min()][col].mean()
    late  = dff[dff["year"] == dff["year"].max()][col].mean()
    return late - early

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🔵 Thinness",   f"{dff[dff['year']==dff['year'].max()]['thinness_pct'].mean():.1f}%",   f"{delta_kpi('thinness_pct'):+.1f}% vs start")
k2.metric("🔴 Obesity",    f"{dff[dff['year']==dff['year'].max()]['obesity_pct'].mean():.1f}%",    f"{delta_kpi('obesity_pct'):+.1f}% vs start")
k3.metric("🟠 Overweight", f"{dff[dff['year']==dff['year'].max()]['overweight_pct'].mean():.1f}%", f"{delta_kpi('overweight_pct'):+.1f}% vs start")
k4.metric("🌐 Countries",  f"{dff['country'].nunique()}", "in selection")
k5.metric("📅 Years",      f"{dff['year'].nunique()}", f"{year_range[0]}–{year_range[1]}")

st.markdown("---")

st.markdown(f'<p class="section-header">📍 Global Distribution & Trends — {indicator.replace("_pct","").title()}</p>', unsafe_allow_html=True)
map_col, trend_col = st.columns([6, 4])

with map_col:
    map_year = st.slider("Map Year", year_range[0], year_range[1], year_range[1], key="map_yr")
    map_df = dff[dff["year"] == map_year].groupby(["country","country_name"])[indicator].mean().reset_index()
    fig_map = px.choropleth(
        map_df, locations="country", locationmode="ISO-3",
        color=indicator, color_continuous_scale=ind_colors[indicator],
        hover_name="country_name", hover_data={indicator: ":.1f", "country": True},
        labels={indicator: ind_labels[indicator]},
        title=f"{ind_labels[indicator]} by Country — {map_year}"
    )

    all_iso3 = [c.alpha_3 for c in pycountry.countries if hasattr(c, "alpha_3")]
    base_trace = go.Choropleth(
        locations=all_iso3,
        z=[0] * len(all_iso3),
        locationmode="ISO-3",
        colorscale=[[0, "#E8ECEF"], [1, "#E8ECEF"]],
        showscale=False,
        hoverinfo="skip",
        marker_line_color="white",
        marker_line_width=0.2,
        zmin=0,
        zmax=1,
    )
    fig_map.add_trace(base_trace)
    fig_map.data = fig_map.data[-1:] + fig_map.data[:-1]

    fig_map.update_geos(
        scope="world",
        projection_type="natural earth",
        showcoastlines=True,
        coastlinecolor="gray",
        showland=False,
        showocean=True,
        oceancolor="lightblue",
    )
    fig_map.update_traces(selector=dict(type="choropleth"), marker_line_width=0.2, marker_line_color="black")
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=350,
        coloraxis_colorbar=dict(title=ind_labels[indicator], thickness=12))
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("**Note:** Colored regions show UNICEF data availability. Blank areas indicate countries not covered in this dataset. This is a global sample of 194 territories with complete nutrition data.")

with trend_col:
    trend_df = dff.groupby(["year", "income_group"])[indicator].mean().reset_index()
    fig_trend = px.line(trend_df, x="year", y=indicator, color="income_group", markers=True,
        color_discrete_map={"Low": "#E24B4A", "Lower Middle": "#EF9F27", "Upper Middle": "#378ADD", "High": "#3B6D11"},
        labels={indicator: ind_labels[indicator], "year": "Year", "income_group": "Income Group"},
        title=f"Trend by Income Group ({year_range[0]}–{year_range[1]})")
    fig_trend.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(title="Income Group", orientation="h", y=-0.3))
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown('<p class="section-header">🔬 Driver Analysis — The Double Burden of Malnutrition</p>', unsafe_allow_html=True)
scatter_col, box_col = st.columns([5, 5])

with scatter_col:
    sample_df = dff.sample(min(2000, len(dff)), random_state=42)
    fig_scatter = px.scatter(sample_df, x="obesity_pct", y="thinness_pct",
        color="un_region", size="overweight_pct", size_max=12, opacity=0.6,
        hover_name="country_name",
        hover_data={"year": True, "thinness_pct": ":.1f", "obesity_pct": ":.1f", "overweight_pct": ":.1f"},
        labels={"obesity_pct": "Obesity (%)", "thinness_pct": "Thinness (%)", "un_region": "UN Region", "overweight_pct": "Overweight (%)"},
        title="Thinness vs. Obesity (bubble = overweight size)")
    z  = np.polyfit(dff["obesity_pct"], dff["thinness_pct"], 1)
    xr = np.linspace(dff["obesity_pct"].min(), dff["obesity_pct"].max(), 100)
    fig_scatter.add_trace(go.Scatter(x=xr, y=np.poly1d(z)(xr), mode="lines", name="Trend",
        line=dict(dash="dash", color="black", width=1.5)))
    fig_scatter.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)
    r = np.corrcoef(dff["obesity_pct"], dff["thinness_pct"])[0, 1]
    st.markdown(f'<div class="insight-box">📊 Pearson r = {r:.3f} — Strong negative correlation. As obesity rises, thinness declines, reflecting the <b>nutrition transition</b> across income levels.</div>', unsafe_allow_html=True)

with box_col:
    region_medians = dff.groupby("un_region")[indicator].median().sort_values(ascending=False)
    fig_box = px.box(dff, x="un_region", y=indicator,
        category_orders={"un_region": region_medians.index.tolist()},
        color="un_region", points=False,
        labels={indicator: ind_labels[indicator], "un_region": "UN Region"},
        title=f"{ind_labels[indicator]} Distribution by Region")
    fig_box.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_box, use_container_width=True)
    top_region = region_medians.index[0]
    st.markdown(f'<div class="insight-box">🌏 Highest median: <b>{top_region}</b> ({region_medians.iloc[0]:.1f}%). Lowest: <b>{region_medians.index[-1]}</b> ({region_medians.iloc[-1]:.1f}%).</div>', unsafe_allow_html=True)

st.markdown('<p class="section-header">📈 Regression Findings — Key Drivers of Child Thinness (HC3 Robust OLS)</p>', unsafe_allow_html=True)
reg_col, interpret_col = st.columns([5, 5])

with reg_col:
    params = params_s.drop("Intercept")
    pvals  = pvals_s.drop("Intercept")
    ci     = ci_df.drop("Intercept")

    sig_mask = pvals < 0.05
    coef_df  = pd.DataFrame({
        "Coefficient": params[sig_mask],
        "CI_low":      ci.loc[sig_mask, 0],
        "CI_high":     ci.loc[sig_mask, 1],
        "p_value":     pvals[sig_mask]
    }).sort_values("Coefficient")

    label_map = {
        "obesity_pct": "Obesity prevalence (%)",
        "year": "Year (time trend)",
        "C(income_group, Treatment('High'))[T.Low]": "Income: Low vs High",
        "C(income_group, Treatment('High'))[T.Lower Middle]": "Income: Lower-Middle vs High",
        "C(income_group, Treatment('High'))[T.Upper Middle]": "Income: Upper-Middle vs High",
        "C(un_region, Treatment('WE'))[T.EAP]": "Region: East Asia-Pacific",
        "C(un_region, Treatment('WE'))[T.EECA]": "Region: Eastern Europe-CA",
        "C(un_region, Treatment('WE'))[T.ESA]": "Region: Eastern-Southern Africa",
        "C(un_region, Treatment('WE'))[T.LAC]": "Region: Latin America",
        "C(un_region, Treatment('WE'))[T.MENA]": "Region: Middle East-N.Africa",
        "C(un_region, Treatment('WE'))[T.NA]": "Region: North America",
        "C(un_region, Treatment('WE'))[T.SA]": "Region: South Asia",
        "C(un_region, Treatment('WE'))[T.WCA]": "Region: West-Central Africa",
    }
    coef_df.index = [label_map.get(i, i) for i in coef_df.index]
    colors_bar = ["#E24B4A" if c < 0 else "#3B6D11" for c in coef_df["Coefficient"]]

    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        x=coef_df["Coefficient"], y=coef_df.index,
        orientation="h", marker_color=colors_bar,
        error_x=dict(type="data", symmetric=False,
            array=coef_df["CI_high"] - coef_df["Coefficient"],
            arrayminus=coef_df["Coefficient"] - coef_df["CI_low"],
            color="gray", thickness=1.5),
        name="Coefficient"
    ))
    fig_coef.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
    fig_coef.update_layout(
        title=f"Significant Drivers of Thinness (p < 0.05) | R² = {ols_model.rsquared:.3f}",
        xaxis_title="Coefficient (effect on Thinness %)",
        height=420, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    st.plotly_chart(fig_coef, use_container_width=True)

with interpret_col:
    st.markdown("#### Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²",      f"{ols_model.rsquared:.3f}")
    m2.metric("Adj. R²", f"{ols_model.rsquared_adj:.3f}")
    m3.metric("Obs.",    f"{int(ols_model.nobs):,}")

    st.markdown("#### Key Insights")
    sa_coef  = params_s.get("C(un_region, Treatment('WE'))[T.SA]", 0)
    low_coef = params_s.get("C(income_group, Treatment('High'))[T.Low]", 0)
    obe_coef = params_s.get("obesity_pct", 0)
    yr_coef  = params_s.get("year", 0)

    for ins in [
        f"🔴 <b>South Asia</b> has the highest thinness burden: +{sa_coef:.1f}% above Western Europe after controlling for other factors.",
        f"💰 <b>Low-income countries</b> show +{low_coef:.1f}% higher thinness than high-income countries — income is the strongest structural predictor.",
        f"⚖️ Each 1% increase in <b>obesity prevalence</b> is associated with {obe_coef:.2f}% lower thinness — the double-burden trade-off.",
        f"📅 <b>Time trend</b>: Thinness declines by {abs(yr_coef):.3f}% per year globally — progress is real but uneven.",
    ]:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

    st.markdown("#### References")
    st.caption(
        "• Black et al. (2013). Maternal and child undernutrition. The Lancet, 382.\n"
        "• Popkin et al. (2020). Double burden of malnutrition. The Lancet, 395.\n"
        "• Victora et al. (2010). Consequences for adult health. The Lancet, 371.\n"
        "• UNICEF (2023). State of the World's Children."
    )

st.markdown('<p class="section-header">🏳️ Country Comparison</p>', unsafe_allow_html=True)
top_n     = st.slider("Show top/bottom N countries", 5, 20, 10)
comp_year = st.select_slider("Comparison Year", options=sorted(dff["year"].unique()), value=2022)

comp_df          = dff[dff["year"] == comp_year].sort_values("thinness_pct", ascending=False)
top_countries    = comp_df.head(top_n)[["country", "thinness_pct", "obesity_pct", "income_group", "un_region"]]
bottom_countries = comp_df.tail(top_n).sort_values("thinness_pct")[["country", "thinness_pct", "obesity_pct", "income_group", "un_region"]]

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**🔴 Highest Thinness Countries — {comp_year}**")
    fig_top = px.bar(top_countries, x="thinness_pct", y="country", orientation="h",
        color="un_region", text="thinness_pct",
        labels={"thinness_pct": "Thinness (%)", "country": ""},
        title=f"Top {top_n} Countries by Thinness")
    fig_top.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_top.update_layout(height=350, showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_top, use_container_width=True)

with c2:
    st.markdown(f"**🟢 Lowest Thinness Countries — {comp_year}**")
    fig_bot = px.bar(bottom_countries, x="thinness_pct", y="country", orientation="h",
        color="un_region", text="thinness_pct",
        labels={"thinness_pct": "Thinness (%)", "country": ""},
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title=f"Bottom {top_n} Countries by Thinness")
    fig_bot.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bot.update_layout(height=350, showlegend=True, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_bot, use_container_width=True)

st.markdown("---")
st.markdown(
    '<p class="footer-note">SDG 3 · Child Nutrition Dashboard | '
    'Data: UNICEF Global Database (August 2025) | '
    'West Visayas State University — Analytics Techniques and Tools Finals ALA | '
    'Built with Streamlit & Plotly</p>',
    unsafe_allow_html=True
)