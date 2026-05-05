import json
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="UK Electricity Price Forecast",
    page_icon="⚡",
    layout="wide",
)

FORECAST_FILE = "city_forecasts.json"

COLORS = ["#2563eb", "#d97706", "#16a34a", "#dc2626",
          "#7c3aed", "#0891b2", "#db2777", "#65a30d"]

@st.cache_data(ttl=300)
def load_forecasts():
    if not os.path.exists(FORECAST_FILE):
        return None
    with open(FORECAST_FILE, encoding="utf-8") as f:
        return json.load(f)

data = load_forecasts()

if data is None:
    st.error("city_forecasts.json not found. Run the notebook first to generate forecasts.")
    st.stop()

cities = {c["name"]: c for c in data["cities"]}
city_names = list(cities.keys())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ UK Electricity\nPrice Forecast")
    st.caption(f"📅 {data['pred_date_str']}")
    st.caption(f"Generated: {data['generated_at']}")
    st.caption(f"Model MAE: {data['model_mae']} EUR/MWh · R²: {data['model_r2']}")
    st.divider()

    st.markdown("### Select Location")
    menu_options = ["Overview — All Cities"] + city_names
    selected_menu = st.radio(
        label="Cities",
        options=menu_options,
        label_visibility="collapsed",
    )
    st.divider()

    # Mini summary in sidebar for quick comparison
    st.markdown("### Quick Stats")
    for name in city_names:
        c = cities[name]
        st.markdown(
            f"**{name}** — {c['mean_price']:.1f} €/MWh  \n"
            f"<small>Peak {c['peak_price']:.0f} @ {c['peak_hour']:02d}:00</small>",
            unsafe_allow_html=True,
        )

# ── Main area ─────────────────────────────────────────────────────────────────
if selected_menu == "Overview — All Cities":
    st.title("UK Electricity Price Forecast")
    st.caption(f"Forecast for **{data['pred_date_str']}**")

    # Summary metric cards
    cols = st.columns(len(city_names))
    for col, name in zip(cols, city_names):
        c = cities[name]
        col.metric(
            label=name,
            value=f"{c['mean_price']:.1f} €",
            delta=f"Peak {c['peak_price']:.0f} @ {c['peak_hour']:02d}:00",
            delta_color="inverse",
        )

    st.divider()

    # All cities line chart
    fig = go.Figure()
    fig.add_vrect(x0=7, x1=10, fillcolor="#d97706", opacity=0.07,
                  line_width=0, annotation_text="AM peak", annotation_position="top left")
    fig.add_vrect(x0=16, x1=20, fillcolor="#dc2626", opacity=0.07,
                  line_width=0, annotation_text="PM peak", annotation_position="top left")

    for i, name in enumerate(city_names):
        c = cities[name]
        fig.add_trace(go.Scatter(
            x=c["hours"], y=c["prices"],
            mode="lines+markers",
            name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{name}</b><br>%{{x:02d}}:00 — %{{y:.1f}} EUR/MWh<extra></extra>",
        ))

    fig.update_layout(
        title=f"Hourly Price Forecast — {data['pred_date_str']}",
        xaxis=dict(title="Hour of day", tickvals=list(range(0, 24, 2)),
                   ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]),
        yaxis=dict(title="EUR / MWh"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=480,
    )
    st.plotly_chart(fig, width='stretch')

    # City comparison bar chart
    means = {n: cities[n]["mean_price"] for n in city_names}
    uk_avg = sum(means.values()) / len(means)
    bar_fig = go.Figure(go.Bar(
        x=list(means.keys()),
        y=list(means.values()),
        marker_color=COLORS[:len(city_names)],
        opacity=0.8,
        hovertemplate="%{x}: %{y:.1f} EUR/MWh<extra></extra>",
    ))
    bar_fig.add_hline(
        y=uk_avg, line_dash="dash", line_color="#dc2626",
        annotation_text=f"UK avg {uk_avg:.1f}",
        annotation_position="bottom right",
    )
    bar_fig.update_layout(
        title="Mean Day-Ahead Price by City",
        yaxis_title="EUR / MWh",
        height=320,
    )
    st.plotly_chart(bar_fig, width='stretch')

    # Full data table
    st.divider()
    st.subheader("Raw Data")
    rows = []
    for name in city_names:
        c = cities[name]
        for h, p, t, w, s in zip(c["hours"], c["prices"], c["temps"], c["winds"], c["solars"]):
            rows.append({
                "City": name, "Hour": f"{h:02d}:00",
                "Price (EUR/MWh)": p, "Temp (°C)": t,
                "Wind (km/h)": w, "Solar (W/m²)": s,
            })
    df = pd.DataFrame(rows)
    city_filter = st.multiselect("Filter cities", city_names, default=city_names)
    filtered = df[df["City"].isin(city_filter)].reset_index(drop=True)
    st.dataframe(filtered, width='stretch', height=400)
    st.download_button("Download CSV", filtered.to_csv(index=False), "forecasts.csv", "text/csv")

else:
    # ── City detail dashboard ──────────────────────────────────────────────────
    name = selected_menu
    c = cities[name]
    color = COLORS[city_names.index(name) % len(COLORS)]
    hours = c["hours"]

    st.title(f"⚡ {name}")
    st.caption(f"Forecast for **{data['pred_date_str']}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean Price",  f"{c['mean_price']:.1f} EUR/MWh")
    m2.metric("Peak Price",  f"{c['peak_price']:.1f} EUR/MWh", f"{c['peak_hour']:02d}:00")
    m3.metric("Low Price",   f"{c['low_price']:.1f} EUR/MWh",  f"{c['low_hour']:02d}:00")
    m4.metric("Day Range",   f"{c['day_range']:.1f} EUR/MWh")

    st.divider()

    # Price + temperature dual-axis
    fig2 = go.Figure()
    fig2.add_vrect(x0=7, x1=10, fillcolor="#d97706", opacity=0.07, line_width=0,
                   annotation_text="AM peak", annotation_position="top left")
    fig2.add_vrect(x0=16, x1=20, fillcolor="#dc2626", opacity=0.07, line_width=0,
                   annotation_text="PM peak", annotation_position="top left")
    fig2.add_trace(go.Scatter(
        x=hours, y=c["prices"], name="Price (EUR/MWh)",
        line=dict(color=color, width=2.5),
        hovertemplate="%{x:02d}:00 — %{y:.1f} EUR/MWh<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=hours, y=c["temps"], name="Temp (°C)",
        yaxis="y2",
        line=dict(color="#dc2626", width=1.5, dash="dot"),
        hovertemplate="%{x:02d}:00 — %{y:.1f}°C<extra></extra>",
    ))
    fig2.update_layout(
        title=f"{name} — Hourly Price & Temperature",
        xaxis=dict(title="Hour", tickvals=list(range(0, 24, 2)),
                   ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]),
        yaxis=dict(title="EUR / MWh"),
        yaxis2=dict(title="°C", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig2, width='stretch')

    # Wind speed bar
    wind_fig = go.Figure(go.Bar(
        x=hours, y=c["winds"],
        marker_color="#16a34a", opacity=0.7,
        hovertemplate="%{x:02d}:00 — %{y:.1f} km/h<extra></extra>",
    ))
    wind_fig.update_layout(
        title=f"{name} — Wind Speed",
        xaxis=dict(title="Hour", tickvals=list(range(0, 24, 2)),
                   ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]),
        yaxis_title="km/h",
        height=280,
    )
    st.plotly_chart(wind_fig, width='stretch')

    # City data table
    st.divider()
    st.subheader("Hourly Data")
    rows = [
        {"Hour": f"{h:02d}:00", "Price (EUR/MWh)": p, "Temp (°C)": t, "Wind (km/h)": w, "Solar (W/m²)": s}
        for h, p, t, w, s in zip(c["hours"], c["prices"], c["temps"], c["winds"], c["solars"])
    ]
    df_city = pd.DataFrame(rows)
    st.dataframe(df_city, width='stretch', height=350)
    st.download_button(
        "Download CSV", df_city.to_csv(index=False),
        f"{name.lower().replace(' ', '_')}_forecast.csv", "text/csv",
    )
