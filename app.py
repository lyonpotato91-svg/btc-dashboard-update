import time
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="BTC Market Dashboard", layout="wide")

# -----------------------------
# Safe requests (NO raise_for_status) + retries
# -----------------------------
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BTC-Dashboard/1.0; +https://streamlit.app)",
    "Accept": "application/json",
}

def safe_get(url, params=None, timeout=15, retries=3):
    """
    - Never raise_for_status (avoid crashing)
    - Retry on 429/5xx with backoff
    - Return (json, status_code)
    """
    last_status = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=DEFAULT_HEADERS)
            last_status = r.status_code
            if r.status_code == 200:
                return r.json(), 200

            # rate limit / transient errors
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep((2 ** i) + random.random())
                continue

            return None, r.status_code
        except Exception:
            time.sleep((2 ** i) + random.random())
            continue
    return None, last_status or -1

# -----------------------------
# Data sources
# -----------------------------
@st.cache_data(ttl=30)
def get_btc_spot_usd():
    """
    Prefer Coinbase/Binance to avoid CoinGecko 429 on Streamlit Cloud.
    Return (price, source)
    """
    # 1) Coinbase
    data, code = safe_get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=12, retries=3)
    if data and "data" in data and "amount" in data["data"]:
        return float(data["data"]["amount"]), "Coinbase"

    # 2) Binance (USDT proxy)
    data, code = safe_get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"}, timeout=12, retries=3)
    if data and "price" in data:
        return float(data["price"]), "Binance"

    # 3) CryptoCompare price
    data, code = safe_get("https://min-api.cryptocompare.com/data/price", params={"fsym": "BTC", "tsyms": "USD"}, timeout=12, retries=3)
    if data and "USD" in data:
        return float(data["USD"]), "CryptoCompare"

    raise RuntimeError("现价数据源全部失败（Coinbase/Binance/CryptoCompare）。")

@st.cache_data(ttl=600)
def get_btc_history_daily(days: int = 365):
    """
    Use CryptoCompare histoday as PRIMARY to avoid CoinGecko 429.
    Return DataFrame: date, price (UTC)
    """
    # CryptoCompare histoday: max ~2000 points without key
    limit = int(min(max(days, 30), 2000)) - 1
    data, code = safe_get(
        "https://min-api.cryptocompare.com/data/v2/histoday",
        params={"fsym": "BTC", "tsym": "USD", "limit": limit},
        timeout=20,
        retries=3,
    )
    if data and data.get("Response") == "Success":
        rows = data["Data"]["Data"]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["price"] = df["close"].astype(float)
        df = df[["date", "price"]].sort_values("date").reset_index(drop=True)
        return df

    raise RuntimeError(f"历史价格失败（CryptoCompare）。HTTP={code}")

@st.cache_data(ttl=600)
def get_fear_greed(limit=200):
    data, code = safe_get("https://api.alternative.me/fng/", params={"limit": limit, "format": "json"}, timeout=20, retries=3)
    if not data or "data" not in data:
        raise RuntimeError(f"恐惧贪婪指数失败（alternative.me）。HTTP={code}")

    df = pd.DataFrame(data["data"])
    df["value"] = df["value"].astype(float)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    return df.sort_values("date").reset_index(drop=True)[["date", "value", "value_classification"]]

# -----------------------------
# Indicators
# -----------------------------
def realized_vol(df_price: pd.DataFrame, window_days: int = 30):
    px = df_price["price"].astype(float).values
    if len(px) < window_days + 2:
        out = df_price.copy()
        out[f"rv{window_days}"] = np.nan
        return out[["date", f"rv{window_days}"]]

    rets = np.diff(np.log(px))
    roll = pd.Series(rets).rolling(window_days).std() * np.sqrt(365)
    out = df_price.iloc[1:].copy()
    out[f"rv{window_days}"] = roll.values
    return out[["date", f"rv{window_days}"]]

def percentile_rank(series: pd.Series, value: float):
    s = series.dropna().astype(float).values
    if len(s) == 0 or np.isnan(value):
        return np.nan
    return float((s < value).mean() * 100.0)

def gauge(value, title, subtitle=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 34}},
        title={"text": f"{title}<br><span style='font-size:12px;color:#888'>{subtitle}</span>"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"thickness": 0.3}}
    ))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def small_card(title, value, footnote):
    st.markdown(
        f"""
        <div style="padding:14px;border:1px solid rgba(255,255,255,0.08);
        border-radius:12px;background:rgba(255,255,255,0.02);">
          <div style="font-size:12px;color:#9aa4b2;margin-bottom:6px;">{title}</div>
          <div style="font-size:22px;font-weight:700;margin-bottom:6px;">{value}</div>
          <div style="font-size:12px;color:#9aa4b2;">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# UI
# -----------------------------
st.title("BTC 市场分析 Dashboard（免本地部署版 / 抗限流）")

_, right = st.columns([3, 1])
with right:
    tf = st.radio("时间范围", ["7d", "30d", "90d", "180d", "1Y"], horizontal=True, index=2)
    auto = st.toggle("自动刷新（60s）", value=False)
    if st.button("手动刷新"):
        st.cache_data.clear()
        st.rerun()

if auto:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()

days_map = {"7d": 7, "30d": 30, "90d": 90, "180d": 180, "1Y": 365}
days = days_map[tf]

# -----------------------------
# Load data
# -----------------------------
try:
    spot, spot_src = get_btc_spot_usd()
    hist = get_btc_history_daily(days=max(365 * 2, days))
    hist_slice = hist[hist["date"] >= (hist["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)

    fng = get_fear_greed(limit=max(200, days + 20))
    fng_slice = fng[fng["date"] >= (fng["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)

    rv30 = realized_vol(hist, 30)
    rv90 = realized_vol(hist, 90)

except Exception as e:
    st.error(f"数据拉取失败：{e}")
    st.stop()

# -----------------------------
# KPIs
# -----------------------------
rv90_last = float(rv90.dropna().iloc[-1]["rv90"]) if rv90["rv90"].notna().any() else np.nan
rv90_pct = percentile_rank(rv90["rv90"], rv90_last)

rv30_last = float(rv30.dropna().iloc[-1]["rv30"]) if rv30["rv30"].notna().any() else np.nan

fng_last = float(fng_slice.iloc[-1]["value"]) if len(fng_slice) else np.nan
fng_class = fng_slice.iloc[-1]["value_classification"] if len(fng_slice) else "N/A"
fng_pct = percentile_rank(fng["value"], fng_last)

# score
score = 50.0
if not np.isnan(rv90_pct):
    score += (50.0 - rv90_pct) * 0.4
if not np.isnan(fng_last):
    score += (25.0 - abs(fng_last - 50.0)) * 0.4
score = float(np.clip(score, 0, 100))

# -----------------------------
# Top row
# -----------------------------
c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

with c1:
    st.plotly_chart(gauge(score, f"{int(round(score))}", "综合市场状态"), use_container_width=True)
    st.caption(f"BTC Spot: ${spot:,.0f}  ·  Source: {spot_src}")

with c2:
    small_card(
        "DVOL（Proxy: RV90）",
        f"{rv90_last*100:.1f}%" if not np.isnan(rv90_last) else "N/A",
        f"90d 分位: {rv90_pct:.0f}%" if not np.isnan(rv90_pct) else "90d 分位: N/A"
    )

with c3:
    small_card(
        "恐惧贪婪指数",
        f"{int(round(fng_last))}（{fng_class}）" if not np.isnan(fng_last) else "N/A",
        f"分位: {fng_pct:.0f}%" if not np.isnan(fng_pct) else "分位: N/A"
    )

with c4:
    small_card(
        "近30日波动（Proxy: RV30）",
        f"{rv30_last*100:.1f}%" if not np.isnan(rv30_last) else "N/A",
        "用于替代 IV 趋势展示"
    )

st.divider()

# -----------------------------
# Charts
# -----------------------------
left1, right1 = st.columns(2)
left2, right2 = st.columns(2)

with left1:
    st.subheader("波动率（RV90） + 价格")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_slice["date"], y=hist_slice["price"], name="BTC Price", yaxis="y2"))
    rv90_s = rv90[rv90["date"] >= hist_slice["date"].min()].copy()
    fig.add_trace(go.Scatter(x=rv90_s["date"], y=rv90_s["rv90"]*100, name="RV90(%)"))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Vol %"),
        yaxis2=dict(title="Price USD", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

with right1:
    st.subheader("恐惧贪婪（FNG） + 价格")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_slice["date"], y=hist_slice["price"], name="BTC Price", yaxis="y2"))
    fig.add_trace(go.Scatter(x=fng_slice["date"], y=fng_slice["value"], name="FNG"))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="FNG (0-100)", range=[0, 100]),
        yaxis2=dict(title="Price USD", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

with left2:
    st.subheader("近30日 Realized Vol（RV30）")
    fig = go.Figure()
    rv30_s = rv30[rv30["date"] >= hist_slice["date"].min()].copy()
    fig.add_trace(go.Scatter(x=rv30_s["date"], y=rv30_s["rv30"]*100, name="RV30(%)"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(title="Vol %"))
    st.plotly_chart(fig, use_container_width=True)

with right2:
    st.subheader("价格走势（选定时间范围）")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_slice["date"], y=hist_slice["price"], name="BTC Price"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(title="Price USD"))
    st.plotly_chart(fig, use_container_width=True)

st.caption("说明：为避免 Streamlit Cloud 被 CoinGecko 429 限流，本版本历史数据采用 CryptoCompare；现价优先 Coinbase→Binance→CryptoCompare。")
