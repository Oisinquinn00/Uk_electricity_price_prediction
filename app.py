import json
import pickle
import warnings
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
warnings.filterwarnings("ignore")
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK Electricity Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: #f8f9fb; }
.main .block-container { padding-top: 1.8rem; max-width: 1180px; }
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
div[data-testid="metric-container"] {
    background: white; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
div[data-testid="metric-container"] label {
    color: #6b7280 !important; font-size: 0.7rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: .07em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.55rem !important; color: #111827 !important;
}
.stButton > button {
    background: #2563eb; color: white; border: none;
    border-radius: 8px; font-weight: 600; padding: 10px 0; width: 100%;
}
.stButton > button:hover { background: #1d4ed8; }
.info-box {
    background: #eff6ff; border-left: 3px solid #2563eb;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-size: .8rem; color: #1e40af; margin-bottom: 14px; line-height: 1.65;
}
.warn-box {
    background: #fffbeb; border-left: 3px solid #d97706;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-size: .78rem; color: #92400e; margin-bottom: 14px; line-height: 1.65;
}
.card {
    background: white; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
</style>
""", unsafe_allow_html=True)
 
 
# =============================================================================
# 1. Load model artifacts (cached — loads only once per session)
# =============================================================================
@st.cache_resource
def load_artifacts():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_cols.json") as f:
        fcols = json.load(f)
    price_hist = pd.read_csv("price_history.csv", parse_dates=["dt"])
    return model, fcols, price_hist
 
 
try:
    model, FCOLS, PRICE_HISTORY = load_artifacts()
except FileNotFoundError as err:
    st.error(
        f"Missing file: **{err.filename}**. "
        "Make sure `rf_model.pkl`, `feature_cols.json`, and "
        "`price_history.csv` are in the same folder as `app.py`."
    )
    st.stop()
 
 
# =============================================================================
# 2. API helpers
# =============================================================================
 
def geocode(query: str) -> dict | None:
    """
    Geocode a city name or UK postcode using the Open-Meteo geocoding API.
    Returns {name, region, country, lat, lon} or None if not found.
    Free, no API key required.
    """
    params = urlencode({"name": query, "count": 1, "language": "en", "format": "json"})
    url = f"https://geocoding-api.open-meteo.com/v1/search?{params}"
    try:
        with urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
        if not results:
            return None
        r = results[0]
        return {
            "name":    r.get("name", query),
            "region":  r.get("admin1", ""),
            "country": r.get("country", ""),
            "lat":     float(r["latitude"]),
            "lon":     float(r["longitude"]),
        }
    except (URLError, KeyError, json.JSONDecodeError, ValueError):
        return None
 
 
def fetch_weather(lat: float, lon: float) -> pd.DataFrame | None:
    """
    Fetch past 8 days + 2 forecast days of hourly weather from Open-Meteo.
    8 past days ensures enough data to compute the 168-hour (7-day) lag features.
    Returns DataFrame [dt, temp, precip, wind, solar, humidity] or None on failure.
    """
    variables = ",".join([
        "temperature_2m", "precipitation",
        "wind_speed_10m", "shortwave_radiation",
        "relative_humidity_2m",
    ])
    params = urlencode({
        "latitude": lat, "longitude": lon,
        "hourly": variables,
        "past_days": 8, "forecast_days": 2,
        "timezone": "Europe/London",
    })
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        with urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        h = data["hourly"]
        df = pd.DataFrame({
            "dt":       pd.to_datetime(h["time"]),
            "temp":     h["temperature_2m"],
            "precip":   h["precipitation"],
            "wind":     h["wind_speed_10m"],
            "solar":    h["shortwave_radiation"],
            "humidity": h["relative_humidity_2m"],
        })
        return df.sort_values("dt").reset_index(drop=True)
    except (URLError, KeyError, json.JSONDecodeError):
        return None
 
 
# =============================================================================
# 3. Feature engineering
# =============================================================================
 
def build_features(price_hist: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price history with live weather and compute all 37 features.
 
    The key design:
      - price_hist contains the last ~200h of UK electricity prices
      - weather contains live hourly weather for the user's location
      - All price lags are >= 24h (no-leakage rule)
      - The last 24 rows with complete features represent today's hours
      - Predicting their target gives tomorrow's 24 hourly prices
 
    Returns a 24-row DataFrame ready for model.predict().
    """
    # Merge: outer join so weather future hours (tomorrow) are included
    df = (price_hist
          .merge(weather, on="dt", how="outer")
          .sort_values("dt")
          .reset_index(drop=True))
 
    # Fill minor weather gaps (DST transitions, API quirks)
    for col in ["temp", "wind", "solar", "humidity", "precip"]:
        df[col] = df[col].ffill().bfill()
 
    # Calendar features
    df["hour"]       = df["dt"].dt.hour
    df["dow"]        = df["dt"].dt.dayofweek
    df["month"]      = df["dt"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_peak"]    = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
 
    # Fourier encoding — smooth cyclic representation
    # Raw integers treat hour 23 and hour 0 as 23 units apart;
    # Fourier makes them adjacent on a circle.
    for k in [1, 2, 3]:
        df[f"sin_h{k}"] = np.sin(2 * np.pi * k * df["hour"] / 24)
        df[f"cos_h{k}"] = np.cos(2 * np.pi * k * df["hour"] / 24)
    for k in [1, 2]:
        df[f"sin_d{k}"] = np.sin(2 * np.pi * k * df["dow"]   / 7)
        df[f"cos_d{k}"] = np.cos(2 * np.pi * k * df["dow"]   / 7)
        df[f"sin_m{k}"] = np.sin(2 * np.pi * k * df["month"] / 12)
        df[f"cos_m{k}"] = np.cos(2 * np.pi * k * df["month"] / 12)
 
    # Price lags — minimum 24h to prevent leakage into the target
    for lag in [24, 48, 168]:
        df[f"p_lag{lag}"] = df["price"].shift(lag)
 
    # Rolling statistics — shifted 24h first so no same-day price is used
    p24 = df["price"].shift(24)
    df["p_roll_mean24"]  = p24.rolling(24).mean()
    df["p_roll_std24"]   = p24.rolling(24).std()
    df["p_roll_mean168"] = p24.rolling(168).mean()
    df["p_roll_min24"]   = p24.rolling(24).min()
    df["p_roll_max24"]   = p24.rolling(24).max()
 
    # Volatility regime: is the market more volatile than its recent baseline?
    std24  = p24.rolling(24).std()
    std168 = p24.rolling(168).std()
    df["vol_regime"] = (std24 / std168.replace(0, np.nan)).fillna(1.0).clip(0, 5)
 
    # Derived weather features
    df["hdd"]        = np.maximum(0, 15.5 - df["temp"])   # heating degree hours
    df["wind_sq"]    = df["wind"] ** 2                     # wind power proxy
    df["temp_lag24"] = df["temp"].shift(24)                # yesterday's temp
    df["wind_lag24"] = df["wind"].shift(24)                # yesterday's wind
 
    # Return the last 24 rows that have all features filled
    return df.dropna(subset=FCOLS).tail(24).copy().reset_index(drop=True)
 
 
def predict_next_day(price_hist: pd.DataFrame,
                     weather: pd.DataFrame) -> pd.DataFrame:
    """
    Run end-to-end prediction.
    Returns 24-row DataFrame: [pred_dt, hour, price, temp, wind, solar, humidity].
    """
    feats = build_features(price_hist, weather)
    if len(feats) < 24:
        return pd.DataFrame()
 
    prices = model.predict(feats[FCOLS])
 
    pred_dts = feats["dt"] + pd.Timedelta(hours=24)
    return pd.DataFrame({
        "pred_dt":  pred_dts,
        "hour":     pred_dts.dt.hour,
        "price":    np.round(prices, 2),
        "temp":     feats["temp"].values,
        "wind":     feats["wind"].values,
        "solar":    feats["solar"].values,
        "humidity": feats["humidity"].values,
    }).reset_index(drop=True)
 
 
# =============================================================================
# 4. Shared Plotly theme
# =============================================================================
 
THEME = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, sans-serif", color="#374151", size=12),
    margin=dict(l=8, r=8, t=40, b=8),
    xaxis=dict(gridcolor="#f3f4f6", showline=False, tickfont_size=11),
    yaxis=dict(gridcolor="#f3f4f6", showline=False, tickfont_size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#1f2937", bordercolor="#374151",
        font=dict(color="white", size=11),
    ),
)
 
 
# =============================================================================
# 5. Sidebar
# =============================================================================
 
with st.sidebar:
    st.markdown(
        "<h2 style='font-size:1.15rem;font-weight:700;color:#111827;"
        "margin-bottom:4px'>⚡ UK Electricity Forecast</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:.75rem;color:#6b7280;margin-bottom:16px'>"
        "Random Forest · 24-hour-ahead · Hourly resolution</p>",
        unsafe_allow_html=True,
    )
    st.divider()
 
    st.markdown("**Your location**")
    location_query = st.text_input(
        "City or UK postcode",
        placeholder="e.g. Manchester, EH1 1YZ",
        label_visibility="collapsed",
    )
    st.caption(
        "Used to fetch live weather from Open-Meteo. "
        "UK electricity price is a national wholesale price — "
        "your location changes the weather inputs only."
    )
 
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚡  Run Forecast", use_container_width=True)
 
    st.divider()
    st.markdown("""
    <div style='font-size:.72rem;color:#6b7280;line-height:1.8'>
    <b style='color:#374151'>Model</b>&nbsp;&nbsp;Random Forest (150 trees)<br>
    <b style='color:#374151'>Features</b>&nbsp;&nbsp;37 engineered inputs<br>
    <b style='color:#374151'>Training</b>&nbsp;&nbsp;Jul 2016 – Dec 2023<br>
    <b style='color:#374151'>Test MAE</b>&nbsp;&nbsp;18.07 EUR/MWh<br>
    <b style='color:#374151'>Test R²</b>&nbsp;&nbsp;0.373<br>
    <b style='color:#374151'>Weather</b>&nbsp;&nbsp;Open-Meteo (free, no key)
    </div>
    """, unsafe_allow_html=True)
 
 
# =============================================================================
# 6. Landing page (shown before user clicks Run)
# =============================================================================
 
st.markdown(
    "<h1 style='font-size:1.55rem;font-weight:700;color:#111827;"
    "margin-bottom:6px'>UK Day-Ahead Electricity Price Forecaster</h1>"
    "<p style='color:#6b7280;font-size:.88rem;margin-bottom:22px'>"
    "Enter a UK location in the sidebar and click <b>Run Forecast</b> "
    "to generate tomorrow's full 24-hour price profile.</p>",
    unsafe_allow_html=True,
)
 
if not run_btn:
    st.markdown("""
    <div class="info-box">
    <b>How it works</b><br>
    1&nbsp; Enter any UK city or postcode in the sidebar<br>
    2&nbsp; Your location is geocoded and live hourly weather is fetched from
    <a href="https://open-meteo.com" target="_blank" style="color:#2563eb">Open-Meteo</a>
    (free, no account needed)<br>
    3&nbsp; Weather is combined with the last 7 days of UK electricity prices
    to build the 37-feature input vector<br>
    4&nbsp; The trained Random Forest outputs a predicted price for every hour
    of tomorrow (00:00 – 23:00)
    </div>
    """, unsafe_allow_html=True)
 
    c1, c2, c3 = st.columns(3)
    for col, icon, title, body in zip(
        [c1, c2, c3],
        ["📍", "🌤️", "📈"],
        ["Enter location", "Live weather fetched", "Full 24h price profile"],
        [
            "Any UK city name or full postcode. "
            "The app resolves it to latitude and longitude automatically.",
            "Temperature, wind speed, solar radiation, humidity and "
            "precipitation fetched in real time for your area.",
            "One predicted price per hour for tomorrow, "
            "with a chart, hourly table and CSV export.",
        ],
    ):
        with col:
            st.markdown(
                f"<div class='card'>"
                f"<div style='font-size:1.6rem;margin-bottom:8px'>{icon}</div>"
                f"<div style='font-weight:600;margin-bottom:4px;color:#111827'>{title}</div>"
                f"<div style='font-size:.78rem;color:#6b7280;line-height:1.55'>{body}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.stop()
 
 
# =============================================================================
# 7. Run the pipeline
# =============================================================================
 
if not location_query.strip():
    st.warning("Please enter a city or postcode in the sidebar.")
    st.stop()
 
# Step 1: Geocode
with st.spinner("Finding your location…"):
    loc = geocode(location_query.strip())
 
if loc is None:
    st.error(
        f"Could not find **{location_query}**. "
        "Try a UK city (e.g. Glasgow) or full postcode (e.g. BS1 4DJ)."
    )
    st.stop()
 
display_loc = (
    f"{loc['name']}"
    f"{', ' + loc['region'] if loc['region'] else ''}"
    f", {loc['country']}"
)
 
# Step 2: Fetch weather
with st.spinner(f"Fetching weather for {display_loc}…"):
    weather_df = fetch_weather(loc["lat"], loc["lon"])
 
if weather_df is None:
    st.error(
        "Could not reach the Open-Meteo weather API. "
        "Check your internet connection and try again."
    )
    st.stop()
 
# Step 3: Predict
with st.spinner("Running the Random Forest model…"):
    forecast = predict_next_day(PRICE_HISTORY, weather_df)
 
if forecast.empty:
    st.error(
        "Not enough price history to build lag features. "
        "Make sure `price_history.csv` contains at least 168 rows."
    )
    st.stop()
 
 
# =============================================================================
# 8. Results
# =============================================================================
 
pred_date  = forecast["pred_dt"].iloc[0].strftime("%A %d %B %Y")
mean_price = float(forecast["price"].mean())
peak_price = float(forecast["price"].max())
low_price  = float(forecast["price"].min())
peak_hour  = int(forecast.loc[forecast["price"].idxmax(), "hour"])
low_hour   = int(forecast.loc[forecast["price"].idxmin(), "hour"])
day_range  = round(peak_price - low_price, 2)
price_chg  = round(mean_price - float(PRICE_HISTORY["price"].tail(24).mean()), 2)
 
st.success(f"✅  Forecast ready for **{display_loc}** — {pred_date}")
 
# KPI strip
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Mean price",   f"{mean_price:.1f}", "EUR/MWh")
with k2: st.metric("Peak price",   f"{peak_price:.1f}", f"at {peak_hour:02d}:00")
with k3: st.metric("Lowest price", f"{low_price:.1f}",  f"at {low_hour:02d}:00")
with k4: st.metric("Day range",    f"{day_range:.1f}",  "EUR/MWh")
with k5:
    sign = "+" if price_chg >= 0 else ""
    st.metric("vs yesterday avg", f"{mean_price:.1f}",
              f"{sign}{price_chg:.1f} EUR/MWh")
 
st.markdown("<br>", unsafe_allow_html=True)
 
# =============================================================================
# 9. Main forecast chart
# =============================================================================
 
st.markdown("#### Predicted hourly prices — tomorrow")
 
hour_labels = [f"{int(h):02d}:00" for h in forecast["hour"]]
 
fig_main = go.Figure()
 
fig_main.add_trace(go.Scatter(
    x=hour_labels, y=forecast["price"],
    fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
    line=dict(color="#2563eb", width=2.2),
    mode="lines", name="Forecast",
    hovertemplate="<b>%{x}</b><br>%{y:.2f} EUR/MWh<extra></extra>",
))
 
fig_main.add_trace(go.Scatter(
    x=[f"{peak_hour:02d}:00"], y=[peak_price],
    mode="markers+text",
    marker=dict(color="#dc2626", size=11),
    text=[f"  Peak {peak_price:.0f}"],
    textposition="middle right",
    textfont=dict(color="#dc2626", size=11),
    showlegend=False,
    hovertemplate=f"Peak: {peak_price:.2f} EUR/MWh<extra></extra>",
))
 
fig_main.add_hline(
    y=mean_price, line_dash="dot", line_color="#9ca3af", line_width=1.2,
    annotation_text=f"Mean {mean_price:.0f}",
    annotation_font=dict(color="#9ca3af", size=10),
    annotation_position="bottom right",
)
 
for x0, x1, fc, label, lc in [
    ("07:00", "09:00", "rgba(217,119,6,.05)",  "Morning peak", "#d97706"),
    ("16:00", "19:00", "rgba(220,38,38,.05)",  "Evening peak", "#dc2626"),
]:
    fig_main.add_vrect(
        x0=x0, x1=x1, fillcolor=fc, layer="below", line_width=0,
        annotation_text=label,
        annotation_font=dict(color=lc, size=9),
        annotation_position="top left",
    )
 
fig_main.update_layout(**THEME, height=380,
                       yaxis_title="EUR / MWh", xaxis_title="Hour of day")
st.plotly_chart(fig_main, use_container_width=True)
 
 
# =============================================================================
# 10. Weather chart + hourly table
# =============================================================================
 
col_wx, col_tbl = st.columns([3, 2])
 
with col_wx:
    st.markdown("#### Weather at your location — tomorrow")
 
    fig_wx = go.Figure()
    fig_wx.add_trace(go.Scatter(
        x=hour_labels, y=forecast["temp"],
        name="Temperature (°C)",
        line=dict(color="#d97706", width=1.8), yaxis="y",
        hovertemplate="%{y:.1f} °C<extra></extra>",
    ))
    fig_wx.add_trace(go.Bar(
        x=hour_labels, y=forecast["wind"],
        name="Wind (km/h)",
        marker_color="rgba(37,99,235,0.38)", yaxis="y2",
        hovertemplate="%{y:.1f} km/h<extra></extra>",
    ))
    fig_wx.add_trace(go.Scatter(
        x=hour_labels, y=forecast["solar"],
        name="Solar (W/m²)",
        line=dict(color="#f59e0b", width=1.2, dash="dot"), yaxis="y3",
        hovertemplate="%{y:.0f} W/m²<extra></extra>",
    ))
    fig_wx.update_layout(
        **THEME, height=300,
        yaxis=dict(title="Temp (°C)", gridcolor="#f3f4f6",
                   titlefont_color="#d97706", tickfont_size=10),
        yaxis2=dict(title="Wind (km/h)", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)",
                    titlefont_color="#2563eb", tickfont_size=10),
        yaxis3=dict(overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)", showticklabels=False),
        legend=dict(orientation="h", y=-0.2, font_size=10),
    )
    st.plotly_chart(fig_wx, use_container_width=True)
 
with col_tbl:
    st.markdown("#### Hourly summary")
 
    def price_band(p: float, mean_p: float) -> str:
        if p >= mean_p * 1.2:  return "🔴 High"
        if p >= mean_p * 0.85: return "🟡 Medium"
        return "🟢 Low"
 
    tbl = pd.DataFrame({
        "Hour":           [f"{int(h):02d}:00" for h in forecast["hour"]],
        "Price EUR/MWh":  forecast["price"].round(2),
        "Level":          [price_band(p, mean_price) for p in forecast["price"]],
        "Temp °C":        forecast["temp"].round(1),
        "Wind km/h":      forecast["wind"].round(1),
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=316)
 
 
# =============================================================================
# 11. Download + disclaimer
# =============================================================================
 
st.divider()
col_dl, col_disc = st.columns([1, 2])
 
with col_dl:
    export = forecast[["pred_dt", "hour", "price",
                        "temp", "wind", "solar", "humidity"]].copy()
    export.columns = ["Datetime", "Hour", "Price_EUR_MWh",
                      "Temp_C", "Wind_kmh", "Solar_Wm2", "Humidity_pct"]
    fname = (
        f"uk_electricity_forecast_"
        f"{forecast['pred_dt'].iloc[0].strftime('%Y-%m-%d')}.csv"
    )
    st.download_button(
        "⬇️  Download forecast as CSV",
        data=export.to_csv(index=False),
        file_name=fname,
        mime="text/csv",
        use_container_width=True,
    )
 
with col_disc:
    st.markdown("""
    <div class='warn-box' style='margin:0'>
    <b>Academic project disclaimer</b> — Produced by a Random Forest trained on
    UK electricity market data 2016–2023. Test MAE = 18.07 EUR/MWh, R² = 0.373.
    Not suitable for commercial trading decisions.
    The model cannot predict sudden market shocks or policy changes.
