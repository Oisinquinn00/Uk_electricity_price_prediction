import io
import json
import pickle
import warnings
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="UK Electricity Forecast",
    page_icon=":zap:",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#e5e7eb", "axes.grid": True,
    "grid.color": "#f3f4f6", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlepad": 10,
})

BLUE  = "#2563eb"
AMBER = "#d97706"
RED   = "#dc2626"
GREY  = "#9ca3af"


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
        "Missing file: " + str(err.filename) + ". "
        "Ensure rf_model.pkl, feature_cols.json and price_history.csv "
        "are in the repo root."
    )
    st.stop()


def geocode(query):
    params = urlencode({"name": query, "count": 1, "language": "en", "format": "json"})
    url = "https://geocoding-api.open-meteo.com/v1/search?" + params
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
    except Exception:
        return None


def fetch_weather(lat, lon):
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
    url = "https://api.open-meteo.com/v1/forecast?" + params
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
    except Exception:
        return None


def build_features(price_hist, weather):
    df = (price_hist
          .merge(weather, on="dt", how="outer")
          .sort_values("dt")
          .reset_index(drop=True))

    for col in ["temp", "wind", "solar", "humidity", "precip"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().bfill().fillna(0.0)

    df["hour"]       = df["dt"].dt.hour
    df["dow"]        = df["dt"].dt.dayofweek
    df["month"]      = df["dt"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_peak"]    = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    for k in [1, 2, 3]:
        df["sin_h" + str(k)] = np.sin(2 * np.pi * k * df["hour"] / 24)
        df["cos_h" + str(k)] = np.cos(2 * np.pi * k * df["hour"] / 24)
    for k in [1, 2]:
        df["sin_d" + str(k)] = np.sin(2 * np.pi * k * df["dow"]   / 7)
        df["cos_d" + str(k)] = np.cos(2 * np.pi * k * df["dow"]   / 7)
        df["sin_m" + str(k)] = np.sin(2 * np.pi * k * df["month"] / 12)
        df["cos_m" + str(k)] = np.cos(2 * np.pi * k * df["month"] / 12)

    for lag in [24, 48, 168]:
        df["p_lag" + str(lag)] = df["price"].shift(lag)

    p24 = df["price"].shift(24)
    df["p_roll_mean24"]  = p24.rolling(24).mean()
    df["p_roll_std24"]   = p24.rolling(24).std()
    df["p_roll_mean168"] = p24.rolling(168).mean()
    df["p_roll_min24"]   = p24.rolling(24).min()
    df["p_roll_max24"]   = p24.rolling(24).max()

    std24  = p24.rolling(24).std()
    std168 = p24.rolling(168).std()
    df["vol_regime"] = (std24 / std168.replace(0, np.nan)).fillna(1.0).clip(0, 5)

    df["hdd"]        = np.maximum(0, 15.5 - df["temp"])
    df["wind_sq"]    = df["wind"] ** 2
    df["temp_lag24"] = df["temp"].shift(24)
    df["wind_lag24"] = df["wind"].shift(24)

    return df.dropna(subset=FCOLS).tail(24).copy().reset_index(drop=True)


def predict_next_day(price_hist, weather):
    feats = build_features(price_hist, weather)
    if len(feats) < 24:
        return pd.DataFrame()
    prices   = model.predict(feats[FCOLS])
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


# Sidebar
with st.sidebar:
    st.markdown(
        "<h2 style='font-size:1.15rem;font-weight:700;color:#111827;"
        "margin-bottom:4px'>UK Electricity Forecast</h2>"
        "<p style='font-size:.75rem;color:#6b7280;margin-bottom:16px'>"
        "Random Forest | 24-hour-ahead | Hourly</p>",
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
        "Used to fetch live weather. UK electricity price is a national "
        "wholesale price -- location affects weather inputs only."
    )
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Forecast", use_container_width=True)
    st.divider()
    st.markdown(
        "<div style='font-size:.72rem;color:#6b7280;line-height:1.8'>"
        "<b style='color:#374151'>Model</b>&nbsp;&nbsp;Random Forest (150 trees)<br>"
        "<b style='color:#374151'>Features</b>&nbsp;&nbsp;37 engineered inputs<br>"
        "<b style='color:#374151'>Training</b>&nbsp;&nbsp;Jul 2016 - Dec 2023<br>"
        "<b style='color:#374151'>Test MAE</b>&nbsp;&nbsp;18.07 EUR/MWh<br>"
        "<b style='color:#374151'>Test R2</b>&nbsp;&nbsp;0.373<br>"
        "<b style='color:#374151'>Weather API</b>&nbsp;&nbsp;Open-Meteo (free)"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<h1 style='font-size:1.55rem;font-weight:700;color:#111827;"
    "margin-bottom:6px'>UK Day-Ahead Electricity Price Forecaster</h1>"
    "<p style='color:#6b7280;font-size:.88rem;margin-bottom:22px'>"
    "Enter a UK location in the sidebar and click Run Forecast.</p>",
    unsafe_allow_html=True,
)

if not run_btn:
    st.markdown(
        "<div class='info-box'>"
        "<b>How it works</b><br>"
        "1 Enter any UK city or postcode in the sidebar<br>"
        "2 The app geocodes your location and fetches live hourly weather "
        "from Open-Meteo (free, no account needed)<br>"
        "3 Weather is combined with the last 7 days of UK electricity prices "
        "to build the 37-feature input vector<br>"
        "4 The trained Random Forest outputs a predicted price for every hour "
        "of tomorrow"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if not location_query.strip():
    st.warning("Please enter a city or postcode in the sidebar.")
    st.stop()

with st.spinner("Finding your location..."):
    loc = geocode(location_query.strip())

if loc is None:
    st.error("Could not find '" + location_query + "'. Try a UK city or full postcode.")
    st.stop()

display_loc = (
    loc["name"]
    + (", " + loc["region"] if loc["region"] else "")
    + ", " + loc["country"]
)

with st.spinner("Fetching weather for " + display_loc + "..."):
    weather_df = fetch_weather(loc["lat"], loc["lon"])

if weather_df is None:
    st.error("Could not reach the Open-Meteo weather API.")
    st.stop()

with st.spinner("Running the Random Forest model..."):
    forecast = predict_next_day(PRICE_HISTORY, weather_df)

if forecast.empty:
    st.error("Not enough price history to build lag features.")
    st.stop()

pred_date  = forecast["pred_dt"].iloc[0].strftime("%A %d %B %Y")
mean_price = float(forecast["price"].mean())
peak_price = float(forecast["price"].max())
low_price  = float(forecast["price"].min())
peak_hour  = int(forecast.loc[forecast["price"].idxmax(), "hour"])
low_hour   = int(forecast.loc[forecast["price"].idxmin(), "hour"])
day_range  = round(peak_price - low_price, 2)
price_chg  = round(mean_price - float(PRICE_HISTORY["price"].tail(24).mean()), 2)

st.success("Forecast ready for " + display_loc + " -- " + pred_date)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Mean price", str(round(mean_price, 1)) + " EUR/MWh")
with k2:
    st.metric("Peak price", str(round(peak_price, 1)) + " EUR/MWh",
              "at " + str(peak_hour).zfill(2) + ":00")
with k3:
    st.metric("Lowest price", str(round(low_price, 1)) + " EUR/MWh",
              "at " + str(low_hour).zfill(2) + ":00")
with k4:
    st.metric("Day range", str(day_range) + " EUR/MWh")
with k5:
    sign = "+" if price_chg >= 0 else ""
    st.metric("vs yesterday avg",
              str(round(mean_price, 1)) + " EUR/MWh",
              sign + str(price_chg) + " EUR/MWh")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Predicted hourly prices -- tomorrow")

hours       = list(range(24))
hour_labels = [str(h).zfill(2) + ":00" for h in hours]
prices      = forecast["price"].tolist()

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.fill_between(hours, prices, alpha=0.08, color=BLUE)
ax.plot(hours, prices, color=BLUE, linewidth=2.2, marker="o",
        markersize=4, label="Forecast price")
ax.plot(peak_hour, peak_price, "o", color=RED, markersize=10, zorder=5)
ax.annotate(
    "Peak " + str(round(peak_price, 1)),
    xy=(peak_hour, peak_price),
    xytext=(peak_hour + (1 if peak_hour < 20 else -4), peak_price + 3),
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    fontsize=10, color=RED,
)
ax.axhline(mean_price, color=GREY, linewidth=1.2, linestyle="--",
           label="Mean " + str(round(mean_price, 1)))
ax.axvspan(7,  10, alpha=0.05, color=AMBER)
ax.axvspan(16, 20, alpha=0.05, color=RED)
ax.set_xticks(hours)
ax.set_xticklabels(hour_labels, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("EUR / MWh")
ax.set_title("Next-Day Hourly Price Forecast -- " + pred_date)
ax.legend(fontsize=10, loc="upper left")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

col_wx, col_tbl = st.columns([3, 2])

with col_wx:
    st.markdown("#### Weather at your location -- tomorrow")
    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    ax2.plot(hours, forecast["temp"].tolist(), color=AMBER, linewidth=1.8,
             label="Temp (C)")
    ax2.set_ylabel("Temperature (C)", color=AMBER)
    ax2.tick_params(axis="y", labelcolor=AMBER)
    ax3 = ax2.twinx()
    ax3.bar(hours, forecast["wind"].tolist(), color=BLUE, alpha=0.35,
            label="Wind (km/h)")
    ax3.set_ylabel("Wind (km/h)", color=BLUE)
    ax3.tick_params(axis="y", labelcolor=BLUE)
    ax3.spines["right"].set_visible(True)
    ax2.set_xticks(hours[::2])
    ax2.set_xticklabels(hour_labels[::2], rotation=40, ha="right", fontsize=9)
    ax2.set_title("Temperature and Wind Speed")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

with col_tbl:
    st.markdown("#### Hourly summary")

    def price_band(p, mean_p):
        if p >= mean_p * 1.2:
            return "High"
        if p >= mean_p * 0.85:
            return "Medium"
        return "Low"

    tbl = pd.DataFrame({
        "Hour":          hour_labels,
        "Price EUR/MWh": forecast["price"].round(2).tolist(),
        "Level":         [price_band(p, mean_price) for p in forecast["price"]],
        "Temp C":        forecast["temp"].round(1).tolist(),
        "Wind km/h":     forecast["wind"].round(1).tolist(),
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=320)

st.divider()
col_dl, col_note = st.columns([1, 2])

with col_dl:
    export = pd.DataFrame({
        "Datetime": [
            forecast["pred_dt"].iloc[0].strftime("%Y-%m-%d") + " "
            + str(int(h)).zfill(2) + ":00"
            for h in forecast["hour"]
        ],
        "Hour":          forecast["hour"].tolist(),
        "Price_EUR_MWh": forecast["price"].tolist(),
        "Temp_C":        forecast["temp"].round(1).tolist(),
        "Wind_kmh":      forecast["wind"].round(1).tolist(),
        "Solar_Wm2":     forecast["solar"].round(0).tolist(),
    })
    fname = (
        "uk_electricity_forecast_"
        + forecast["pred_dt"].iloc[0].strftime("%Y-%m-%d")
        + ".csv"
    )
    st.download_button(
        "Download forecast CSV",
        data=export.to_csv(index=False),
        file_name=fname,
        mime="text/csv",
        use_container_width=True,
    )

with col_note:
    st.markdown(
        "<div class='warn-box' style='margin:0'>"
        "<b>Academic project disclaimer</b> -- Produced by a Random Forest "
        "trained on UK electricity market data 2016-2023. "
        "Test MAE = 18.07 EUR/MWh, R2 = 0.373. "
        "Not for commercial trading decisions."
        "</div>",
        unsafe_allow_html=True,
    )
