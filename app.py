from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Sri Lanka Food Price Forecaster",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_food_price_model.joblib"
RESULTS_DIR = BASE_DIR / "results"


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    fi = pd.read_csv(RESULTS_DIR / "feature_importance.csv")
    latest = pd.read_csv(RESULTS_DIR / "latest_commodity_state.csv")
    hist = pd.read_csv(RESULTS_DIR / "commodity_history.csv")
    hist["date"] = pd.to_datetime(hist["date"])
    return comp, fi, latest, hist


if not MODEL_PATH.exists():
    st.error("Model file missing. Run `notebook.ipynb` first.")
    st.stop()

artifact = load_artifact()
comp_df, fi_df, latest_df, hist_df = load_data()
best_m = artifact["metrics"]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def predict_price(row_df: pd.DataFrame) -> float:
    if artifact.get("is_hybrid", False):
        X_proc = artifact["preprocessor"].transform(row_df)
        rf_p = artifact["rf_stage"].predict(X_proc)
        rf_s = artifact["rf_pred_scaler"].transform(rf_p.reshape(-1, 1))
        X_hyb = np.column_stack([X_proc, rf_s])
        val = float(rf_p[0] + artifact["ann_stage"].predict(X_hyb)[0])
    else:
        val = float(artifact["pipeline"].predict(row_df)[0])
    return max(0.0, val)


PALETTE = {
    "navy": "#1E3A5F",
    "teal": "#2A7B88",
    "amber": "#C87D28",
    "slate": "#64748B",
    "silver": "#94A3B8",
    "plum": "#6B4E71",
    "sage": "#4A7C59",
    "grid": "#E2E8F0",
    "ink": "#1E293B",
}

SERIES_COLORS = [
    PALETTE["navy"],
    PALETTE["teal"],
    PALETTE["amber"],
    PALETTE["plum"],
    PALETTE["sage"],
    PALETTE["slate"],
]

CATEGORY_COLORS = {
    "Boosting": PALETTE["navy"],
    "Stacking": PALETTE["teal"],
    "Boosting (Tuned)": "#3B6998",
    "Bagging": PALETTE["slate"],
    "Hybrid": PALETTE["amber"],
}


def format_chart(fig, height=400):
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="#FAFAFA",
        plot_bgcolor="#FFFFFF",
        font=dict(color=PALETTE["ink"], size=12),
        title=dict(font=dict(size=14, color=PALETTE["ink"])),
        margin=dict(l=20, r=30, t=48, b=48),
        xaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"]),
        yaxis=dict(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"]),
        hovermode="x unified",
    )
    return fig


# ------------------------------------------------------------------------------
# SIDEBAR: DIRECT INTERACTIVE FORECAST CONTROLS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("Forecast Inputs")
    st.caption("UN WFP / HDX · Sri Lanka Retail Prices (LKR)")

    provinces = sorted(latest_df["admin1"].unique())
    prov = st.selectbox(
        "Province",
        provinces,
        index=provinces.index("Western") if "Western" in provinces else 0,
    )

    markets = sorted(latest_df.loc[latest_df["admin1"] == prov, "market"].unique())
    market = st.selectbox("Market", markets)

    mkt_rows = latest_df[(latest_df["admin1"] == prov) & (latest_df["market"] == market)]
    commodities = sorted(mkt_rows["commodity"].unique())
    commodity = st.selectbox(
        "Commodity",
        commodities,
        index=commodities.index("Rice (white)") if "Rice (white)" in commodities else 0,
    )

    ref = mkt_rows[mkt_rows["commodity"] == commodity].iloc[0]
    unit = ref["unit"]
    category = ref["category"]
    last_date = pd.to_datetime(ref["date"])

    month_name = st.selectbox("Forecast Month", MONTHS, index=(last_date.month % 12))
    month_num = MONTHS.index(month_name) + 1

    st.divider()
    st.subheader(f"Recent Prices (LKR / {unit})")
    st.caption(f"Auto-filled from latest record ({last_date.strftime('%Y-%m-%d')})")

    lag1 = st.number_input("Last month (t-1)", min_value=1.0, max_value=25000.0, value=float(round(ref["price"], 2)), step=5.0)
    lag2 = st.number_input("2 months ago (t-2)", min_value=1.0, max_value=25000.0, value=float(round(ref["lag_1m_price"], 2)), step=5.0)
    lag3 = st.number_input("3 months ago (t-3)", min_value=1.0, max_value=25000.0, value=float(round(ref["lag_2m_price"], 2)), step=5.0)

    st.divider()
    st.caption(
        f"Model: {artifact['model_name']}  \n"
        f"2025 Holdout RMSE: LKR {best_m['RMSE (LKR)']:.2f}  \n"
        f"2025 Holdout R²: {best_m['R2 Score']:.4f}"
    )

# Build model input
rolling_mean = float(np.mean([lag1, lag2, lag3]))
rolling_std = float(np.std([lag1, lag2, lag3], ddof=1))
momentum = lag1 - lag2

input_df = pd.DataFrame([{
    "lag_1m_price": lag1,
    "lag_2m_price": lag2,
    "lag_3m_price": lag3,
    "rolling_3m_mean": rolling_mean,
    "rolling_3m_std": rolling_std,
    "momentum_1m": momentum,
    "latitude": float(ref["latitude"]),
    "longitude": float(ref["longitude"]),
    "month_sin": np.sin(2 * np.pi * month_num / 12.0),
    "month_cos": np.cos(2 * np.pi * month_num / 12.0),
    "admin1": prov,
    "market": market,
    "category": category,
    "commodity": commodity,
    "unit": unit,
}])

pred_lkr = predict_price(input_df)
diff_lkr = pred_lkr - lag1
diff_pct = (diff_lkr / lag1) * 100.0 if lag1 > 0 else 0.0

# ------------------------------------------------------------------------------
# MAIN HEADER & TABS
# ------------------------------------------------------------------------------
st.title("Sri Lanka Retail Food Price Forecaster")
st.caption(
    "Ensemble Learning & Hybrid Modeling Dashboard · Data: UN WFP / HDX (Dept. of Census & Statistics & HARTI, 2004–2025)"
)

tab_forecast, tab_models, tab_markets = st.tabs([
    "Price Forecast",
    "Model Comparison",
    "Provincial & Commodity Trends",
])

# ==============================================================================
# TAB 1: PRICE FORECAST
# ==============================================================================
with tab_forecast:
    st.subheader(f"{commodity} — {market} ({prov} Province)")
    st.caption(f"Category: {category} · Unit: {unit} · Target month: {month_name}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        label=f"Forecast ({month_name})",
        value=f"LKR {pred_lkr:,.2f}",
        delta=f"{diff_lkr:+.2f} LKR ({diff_pct:+.1f}%)",
        delta_color="inverse",
    )
    m2.metric(label="Last Month (t-1)", value=f"LKR {lag1:,.2f}")
    m3.metric(label="3-Month Average", value=f"LKR {rolling_mean:,.2f}")
    m4.metric(label="3-Month Volatility (Std)", value=f"LKR {rolling_std:,.2f}")

    st.divider()

    # Historical price series + forecasted next-month marker
    sub_hist = hist_df[
        (hist_df["admin1"] == prov) & (hist_df["commodity"] == commodity)
    ].sort_values("date")

    if len(sub_hist) > 1:
        next_dt = sub_hist["date"].max() + pd.DateOffset(months=1)
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Scatter(
                x=sub_hist["date"],
                y=sub_hist["avg_price"],
                mode="lines+markers",
                name=f"{prov} Historical Avg",
                line=dict(color=PALETTE["navy"], width=2.4),
                marker=dict(size=5, color=PALETTE["navy"]),
            )
        )
        fig_hist.add_trace(
            go.Scatter(
                x=[sub_hist["date"].iloc[-1], next_dt],
                y=[sub_hist["avg_price"].iloc[-1], pred_lkr],
                mode="lines+markers",
                name=f"Forecast ({month_name})",
                line=dict(color=PALETTE["amber"], width=2.4, dash="dot"),
                marker=dict(size=9, symbol="diamond", color=PALETTE["amber"]),
            )
        )
        fig_hist.update_layout(
            title=f"{commodity} Retail Price History & Next-Month Forecast ({prov} Province)",
            xaxis_title="Date",
            yaxis_title=f"Price (LKR / {unit})",
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        )
        st.plotly_chart(format_chart(fig_hist, height=420), use_container_width=True)
    else:
        st.info("Not enough historical records to plot a timeline for this province-commodity pair.")

    with st.expander("Inspect engineered features passed to the model"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: MODEL COMPARISON
# ==============================================================================
with tab_models:
    st.subheader("Chronological Holdout Evaluation (March – September 2025)")
    st.caption(
        f"Trained on {artifact.get('train_rows', 0):,} rows (2023–Feb 2025) · "
        f"Tested on {artifact.get('test_rows', 0):,} unseen rows (Mar–Sep 2025)"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best RMSE Model", artifact["model_name"])
    k2.metric("Lowest RMSE", f"LKR {best_m['RMSE (LKR)']:.2f}")
    k3.metric("Lowest MAE", f"LKR {comp_df['MAE (LKR)'].min():.2f}")
    k4.metric("Highest R²", f"{comp_df['R2 Score'].max():.4f}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        s_rmse = comp_df.sort_values("RMSE (LKR)", ascending=False)
        fig_rmse = px.bar(
            s_rmse,
            x="RMSE (LKR)",
            y="Model",
            color="Category",
            orientation="h",
            text="RMSE (LKR)",
            title="Test RMSE in LKR (Lower is Better)",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig_rmse.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        fig_rmse.update_layout(
            yaxis_title="",
            xaxis_range=[0, s_rmse["RMSE (LKR)"].max() * 1.15],
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, title_text=""),
        )
        st.plotly_chart(format_chart(fig_rmse, height=400), use_container_width=True)

    with c2:
        s_r2 = comp_df.sort_values("R2 Score", ascending=True)
        fig_r2 = px.bar(
            s_r2,
            x="R2 Score",
            y="Model",
            color="Category",
            orientation="h",
            text="R2 Score",
            title="Test R² Score (Higher is Better)",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig_r2.update_traces(texttemplate="%{text:.4f}", textposition="outside", cliponaxis=False)
        fig_r2.update_layout(
            yaxis_title="",
            xaxis_range=[max(0.90, s_r2["R2 Score"].min() - 0.01), 1.01],
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, title_text=""),
        )
        st.plotly_chart(format_chart(fig_r2, height=400), use_container_width=True)

    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with st.expander("Summary of model behavior (Bagging vs. Boosting vs. Stacking vs. Hybrid)"):
        st.markdown(
            """
            - **Boosting (XGBoost, CatBoost, LightGBM)** outperformed **Bagging (Random Forest)** on RMSE and R² because sequential trees learn seasonal adjustments and momentum on volatile vegetables (carrots, tomatoes, beans), whereas Random Forest placed 87.9% of its importance on `lag_1m_price` alone.
            - **Stacking (RF + XGB + LGBM → Ridge)** achieved the **lowest MAE (LKR 68.64)** by blending tree ensembles with linear regularization.
            - **Hybrid (RF + ANN)** trained a 2-layer neural network on out-of-fold RF residuals; on tabular lag features with wide LKR price scales across 42 commodities, gradient boosted trees remained sharper than the neural residual stage.
            """
        )

# ==============================================================================
# TAB 3: PROVINCIAL & COMMODITY TRENDS
# ==============================================================================
with tab_markets:
    all_comms = sorted(hist_df["commodity"].unique())

    c_top1, c_top2 = st.columns([1.2, 1])

    with c_top1:
        default_sel = [c for c in ["Rice (white)", "Lentils", "Wheat flour", "Tomatoes"] if c in all_comms]
        picked = st.multiselect("Compare National Monthly Trends (LKR)", all_comms, default=default_sel)
        if picked:
            nat = hist_df[hist_df["commodity"].isin(picked)].groupby(["date", "commodity"], as_index=False)["avg_price"].mean()
            fig_nat = px.line(
                nat,
                x="date",
                y="avg_price",
                color="commodity",
                color_discrete_sequence=SERIES_COLORS,
                title="National Average Retail Price Trajectory (2023–2025)",
                labels={"date": "Date", "avg_price": "Avg Price (LKR)", "commodity": ""},
            )
            fig_nat.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, title_text=""))
            st.plotly_chart(format_chart(fig_nat, height=410), use_container_width=True)

    with c_top2:
        sel_comm = st.selectbox(
            "Compare Across Sri Lanka's 9 Provinces",
            all_comms,
            index=all_comms.index("Rice (white)") if "Rice (white)" in all_comms else 0,
        )
        p_df = (
            hist_df[hist_df["commodity"] == sel_comm]
            .groupby("admin1", as_index=False)["avg_price"].mean()
            .sort_values("avg_price", ascending=True)
        )
        fig_p = px.bar(
            p_df,
            x="avg_price",
            y="admin1",
            orientation="h",
            text="avg_price",
            title=f"Mean Retail Price of {sel_comm} by Province",
            labels={"avg_price": "Average Price (LKR)", "admin1": ""},
        )
        fig_p.update_traces(marker_color=PALETTE["teal"], texttemplate="LKR %{text:,.0f}", textposition="outside", cliponaxis=False)
        fig_p.update_layout(xaxis_range=[0, p_df["avg_price"].max() * 1.18])
        st.plotly_chart(format_chart(fig_p, height=410), use_container_width=True)

    st.divider()
    top_fi = fi_df.head(10).sort_values("Mean_Importance", ascending=True)
    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(y=top_fi["Feature"], x=top_fi["RF_Importance"], name="Random Forest (Bagging)", orientation="h", marker_color=PALETTE["silver"]))
    fig_fi.add_trace(go.Bar(y=top_fi["Feature"], x=top_fi["LGBM_Importance"], name="LightGBM (Boosting)", orientation="h", marker_color=PALETTE["navy"]))
    fig_fi.update_layout(
        barmode="group",
        title="Top 10 Feature Importances — Random Forest vs. LightGBM",
        xaxis_title="Normalized Importance",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    )
    st.plotly_chart(format_chart(fig_fi, height=400), use_container_width=True)
