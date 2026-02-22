import time
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="BTC Market Dashboard", layout="wide")

# =========================================================
# Compatibility helpers (remove Streamlit deprecation warnings)
# =========================================================
def plotly_show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

def dataframe_show(df):
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)

# =========================================================
# Safe requests (NO raise_for_status) + retries
# =========================================================
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BTC-Dashboard/1.0; +https://streamlit.app)",
    "Accept": "application/json",
}

def safe_get(url, params=None, timeout=15, retries=3):
    """
    Return (json, status_code, error_text)
    - Never raise_for_status (avoid crashing)
    - Retry on 429/5xx with backoff
    """
    last_status = None
    last_err = ""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=DEFAULT_HEADERS)
            last_status = r.status_code

            if r.status_code == 200:
                return r.json(), 200, ""

            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep((2 ** i) + random.random())
                continue

            last_err = (r.text or "")[:200]
            return None, r.status_code, last_err

        except Exception as e:
            last_err = str(e)[:200]
            time.sleep((2 ** i) + random.random())
            continue

    return None, last_status or -1, last_err

# =========================================================
# Data sources (avoid CoinGecko to prevent 429 on Cloud)
# =========================================================
@st.cache_data(ttl=30)
def get_btc_spot_usd():
    """
    Return (price, source, http_code)
    """
    # 1) Coinbase
    data, code, _ = safe_get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=12, retries=3)
    if data and "data" in data and "amount" in data["data"]:
        return float(data["data"]["amount"]), "Coinbase", 200
    coinbase_code = code

    # 2) Binance (USDT proxy)
    data, code, _ = safe_get(
        "https://api.binance.com/api/v3/ticker/price",
        params={"symbol": "BTCUSDT"},
        timeout=12,
        retries=3
    )
    if data and "price" in data:
        return float(data["price"]), "Binance", 200
    binance_code = code

    # 3) CryptoCompare price
    data, code, _ = safe_get(
        "https://min-api.cryptocompare.com/data/price",
        params={"fsym": "BTC", "tsyms": "USD"},
        timeout=12,
        retries=3
    )
    if data and "USD" in data:
        return float(data["USD"]), "CryptoCompare", 200
    cc_code = code

    raise RuntimeError(f"现价数据源全部失败：Coinbase HTTP={coinbase_code}, Binance HTTP={binance_code}, CryptoCompare HTTP={cc_code}")

@st.cache_data(ttl=600)
def get_btc_history_daily(days: int = 365):
    """
    CryptoCompare histoday (daily candles)
    Return (df, source, http_code)
    """
    limit = int(min(max(days, 30), 2000)) - 1
    data, code, err = safe_get(
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
        return df, "CryptoCompare", 200

    raise RuntimeError(f"历史价格失败（CryptoCompare）。HTTP={code} err={err}")

@st.cache_data(ttl=600)
def get_fear_greed(limit=200):
    """
    Return (df, source, http_code)
    """
    data, code, err = safe_get(
        "https://api.alternative.me/fng/",
        params={"limit": limit, "format": "json"},
        timeout=20,
        retries=3
    )
    if data and "data" in data:
        df = pd.DataFrame(data["data"])
        df["value"] = df["value"].astype(float)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df = df.sort_values("date").reset_index(drop=True)[["date", "value", "value_classification"]]
        return df, "alternative.me", 200

    raise RuntimeError(f"恐惧贪婪指数失败（alternative.me）。HTTP={code} err={err}")

@st.cache_data(ttl=600)
def get_deribit_vol_index(days: int = 120):
    """
    Deribit volatility index data (BTC)
    Return (df, source, http_code)

    If endpoint is unavailable, raise and caller will fallback to RV30.
    """
    # Common Deribit endpoint:
    # https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC
    data, code, err = safe_get(
        "https://www.deribit.com/api/v2/public/get_volatility_index_data",
        params={"currency": "BTC"},
        timeout=20,
        retries=3
    )
    if data and data.get("result"):
        result = data["result"]
        rows = result.get("data") or result.get("volatility_index_data")
        if rows and isinstance(rows, list) and len(rows) > 5:
            df = pd.DataFrame(rows)
            # expected: timestamp(ms), value
            if "timestamp" in df.columns and "value" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df["iv"] = df["value"].astype(float) / 100.0  # convert to decimal if it's percentage-like
                # Heuristic: if values look like 50-100, treat as % already and divide by 100.
                # If values are 0.5-1.0 range, keep as-is.
                if df["iv"].median() < 2.0:
                    pass
                else:
                    df["iv"] = df["value"].astype(float) / 100.0
                df = df[["date", "iv"]].sort_values("date").reset_index(drop=True)

                # keep only recent N days
                cutoff = df["date"].max() - pd.Timedelta(days=days)
                df = df[df["date"] >= cutoff].reset_index(drop=True)
                return df, "Deribit", 200

    raise RuntimeError(f"Deribit IV 指数失败。HTTP={code} err={err}")

# =========================================================
# Indicators / Analytics
# =========================================================
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

def score_from_metrics(vol_pct, fng_value, band_pos):
    """
    0-100 score:
    - lower vol percentile => higher score
    - FNG near 50 better, extremes worse
    - band position near mid better than extreme top
    """
    score = 50.0
    if vol_pct is not None and not np.isnan(vol_pct):
        score += (50.0 - vol_pct) * 0.4  # +/-20
    if fng_value is not None and not np.isnan(fng_value):
        score += (25.0 - abs(fng_value - 50.0)) * 0.4  # max +10
    if band_pos is not None and not np.isnan(band_pos):
        score += (0.25 - abs(band_pos - 0.5)) * 40.0  # max +10
    return float(np.clip(score, 0, 100))

def build_rainbow_bands(df_price_all: pd.DataFrame):
    """
    Self-contained rainbow-like band:
    log(price) ~ a + b*log(time)
    Build bands using residual std in log space.
    """
    df = df_price_all.dropna().reset_index(drop=True).copy()
    t = np.arange(1, len(df) + 1, dtype=float)
    x = np.log(t)
    y = np.log(df["price"].astype(float).values)

    b = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    a = y.mean() - b * x.mean()
    y_hat = a + b * x
    resid = y - y_hat
    sigma = resid.std()

    ks = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    out = pd.DataFrame({"date": df["date"], "price": df["price"].astype(float)})
    for k in ks:
        out[f"b{k:+.1f}"] = np.exp(y_hat + k * sigma)
    return out

def current_band_position(rainbow_df: pd.DataFrame):
    last = rainbow_df.iloc[-1]
    low = last["b-2.0"]
    high = last["b+2.0"]
    p = last["price"]
    if high <= low:
        return np.nan, "N/A"
    pos = float((p - low) / (high - low))
    zones = [
        (0.00, 0.125, "Band 1（偏低）"),
        (0.125, 0.25, "Band 2（偏低）"),
        (0.25, 0.375, "Band 3（中低）"),
        (0.375, 0.5, "Band 4（中性）"),
        (0.5, 0.625, "Band 5（中性偏高）"),
        (0.625, 0.75, "Band 6（偏高）"),
        (0.75, 0.875, "Band 7（偏高）"),
        (0.875, 1.00, "Band 8（极高）"),
    ]
    label = zones[-1][2]
    for lo, hi, name in zones:
        if pos >= lo and pos < hi:
            label = name
            break
    return pos, label

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

def small_card(title, value, footnote, bar_pct=None):
    bar_html = ""
    if bar_pct is not None and not np.isnan(bar_pct):
        bar_pct = float(np.clip(bar_pct, 0, 100))
        bar_html = f"""
        <div style="margin-top:10px;background:rgba(255,255,255,0.06);border-radius:999px;height:8px;overflow:hidden;">
          <div style="width:{bar_pct:.1f}%;height:8px;background:rgba(255,255,255,0.35);"></div>
        </div>
        <div style="font-size:11px;color:#9aa4b2;margin-top:6px;">分位：{bar_pct:.0f}%</div>
        """

    st.markdown(
        f"""
        <div style="padding:14px;border:1px solid rgba(255,255,255,0.08);
        border-radius:12px;background:rgba(255,255,255,0.02);">
          <div style="font-size:12px;color:#9aa4b2;margin-bottom:6px;">{title}</div>
          <div style="font-size:22px;font-weight:700;margin-bottom:6px;">{value}</div>
          <div style="font-size:12px;color:#9aa4b2;">{footnote}</div>
          {bar_html}
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# UI
# =========================================================
st.title("BTC 市场分析 Dashboard（抗限流 + IV + 彩虹带）")

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

# =========================================================
# Load data
# =========================================================
source_status = []

try:
    spot, spot_src, _ = get_btc_spot_usd()
    source_status.append(("现价 Spot", spot_src, "OK"))

    hist, hist_src, _ = get_btc_history_daily(days=max(365 * 3, days))  # more history for rainbow
    source_status.append(("历史日线", hist_src, "OK"))

    fng, fng_src, _ = get_fear_greed(limit=max(200, days + 30))
    source_status.append(("恐惧贪婪", fng_src, "OK"))

    # Deribit IV (optional). If fails, fallback to RV30 as proxy.
    try:
        iv_df, iv_src, _ = get_deribit_vol_index(days=max(180, days))
        source_status.append(("期权IV指数", iv_src, "OK"))
        iv_is_proxy = False
    except Exception as e:
        iv_df = None
        iv_src = "Proxy: RV30"
        source_status.append(("期权IV指数", iv_src, "FALLBACK"))
        iv_is_proxy = True

except Exception as e:
    st.error(f"数据拉取失败：{e}")
    st.stop()

# Data source status panel
with st.expander("数据源状态（点开查看）", expanded=False):
    df_status = pd.DataFrame(source_status, columns=["模块", "数据源", "状态"])
    dataframe_show(df_status)
    st.caption("如果未来出现 429/403，这里会显示失败模块，方便快速定位。")

# =========================================================
# Slice + metrics
# =========================================================
hist = hist.sort_values("date").reset_index(drop=True)
hist_slice = hist[hist["date"] >= (hist["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)

fng = fng.sort_values("date").reset_index(drop=True)
fng_slice = fng[fng["date"] >= (fng["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)

rv30 = realized_vol(hist, 30)
rv90 = realized_vol(hist, 90)

rv90_last = float(rv90.dropna().iloc[-1]["rv90"]) if rv90["rv90"].notna().any() else np.nan
rv90_pct = percentile_rank(rv90["rv90"], rv90_last)

fng_last = float(fng_slice.iloc[-1]["value"]) if len(fng_slice) else np.nan
fng_class = fng_slice.iloc[-1]["value_classification"] if len(fng_slice) else "N/A"
fng_pct = percentile_rank(fng["value"], fng_last)

# Rainbow
rainbow = build_rainbow_bands(hist)
band_pos, band_label = current_band_position(rainbow)

# IV (Deribit or proxy)
if iv_df is None:
    iv_series = rv30.rename(columns={"rv30": "iv"}).copy()
    iv_series = iv_series.dropna().reset_index(drop=True)
else:
    iv_series = iv_df.copy().dropna().reset_index(drop=True)

iv_slice = iv_series[iv_series["date"] >= (iv_series["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)
iv_last = float(iv_slice.iloc[-1]["iv"]) if len(iv_slice) else np.nan
iv_prev = float(iv_slice.iloc[-2]["iv"]) if len(iv_slice) >= 2 else np.nan
iv_change = (iv_last - iv_prev) if (not np.isnan(iv_last) and not np.isnan(iv_prev)) else np.nan
iv_pct = percentile_rank(iv_series["iv"], iv_last) if len(iv_series) else np.nan

score = score_from_metrics(rv90_pct, fng_last, band_pos)

# =========================================================
# Top KPI row (closer to your reference)
# =========================================================
c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])

with c1:
    plotly_show(gauge(score, f"{int(round(score))}", "综合市场状态"))
    st.caption(f"BTC Spot: ${spot:,.0f}  ·  Source: {spot_src}")

with c2:
    small_card(
        "DVOL（Proxy: RV90）",
        f"{rv90_last*100:.1f}%" if not np.isnan(rv90_last) else "N/A",
        "波动率越高，风险越高",
        bar_pct=rv90_pct
    )

with c3:
    small_card(
        "恐惧贪婪指数",
        f"{int(round(fng_last))}（{fng_class}）" if not np.isnan(fng_last) else "N/A",
        "极端值往往意味着情绪拥挤",
        bar_pct=fng_pct
    )

with c4:
    small_card(
        "彩虹带位置",
        band_label,
        f"位置: {band_pos:.2f}" if not np.isnan(band_pos) else "位置: N/A",
        bar_pct=(band_pos * 100 if not np.isnan(band_pos) else None)
    )

with c5:
    arrow = "▲" if (not np.isnan(iv_change) and iv_change >= 0) else "▼"
    iv_name = "IV（Deribit）" if not iv_is_proxy else "IV（Proxy: RV30）"
    small_card(
        iv_name,
        f"{iv_last*100:.1f}%" if not np.isnan(iv_last) else "N/A",
        f"24h: {arrow} {iv_change*100:+.2f}%" if not np.isnan(iv_change) else "24h: N/A",
        bar_pct=iv_pct
    )

st.divider()

# =========================================================
# Charts (match the 4-panel structure)
# 1) DVOL/RV90 + price
# 2) FNG + price
# 3) Rainbow band (filled)
# 4) IV trend + price
# =========================================================
left1, right1 = st.columns(2)
left2, right2 = st.columns(2)

with left1:
    st.subheader("波动率（RV90） + 价格")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_slice["date"], y=hist_slice["price"], name="BTC Price", yaxis="y2"))
    rv90_s = rv90[rv90["date"] >= hist_slice["date"].min()].copy()
    fig.add_trace(go.Scatter(x=rv90_s["date"], y=rv90_s["rv90"] * 100, name="RV90(%)"))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Vol %"),
        yaxis2=dict(title="Price USD", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    plotly_show(fig)

with right1:
    st.subheader("恐惧贪婪（FNG） + 价格")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_slice["date"], y=hist_slice["price"], name="BTC Price", yaxis="y2"))
    fig.add_trace(go.Scatter(x=fng_slice["date"], y=fng_slice["value"], name="FNG"))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="FNG (0-100)", range=[0, 100]),
        yaxis2=dict(title="Price USD", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    plotly_show(fig)

with left2:
    st.subheader("比特币彩虹带（估值带）")
    # show last ~3 years for readability
    rb = rainbow.copy()
    rb = rb[rb["date"] >= (rb["date"].max() - pd.Timedelta(days=365*3))].reset_index(drop=True)

    fig = go.Figure()

    # Fill bands from low to high (simple layered fill)
    band_keys = ["b-2.0", "b-1.5", "b-1.0", "b-0.5", "b+0.0", "b+0.5", "b+1.0", "b+1.5", "b+2.0"]
    # Start with lowest line
    fig.add_trace(go.Scatter(x=rb["date"], y=rb[band_keys[0]], name="Band low", line=dict(width=1), opacity=0.2))
    # Fill successive bands
    for i in range(1, len(band_keys)):
        fig.add_trace(go.Scatter(
            x=rb["date"],
            y=rb[band_keys[i]],
            name=f"Band {i}",
            line=dict(width=1),
            fill="tonexty",
            opacity=0.12
        ))

    fig.add_trace(go.Scatter(x=rb["date"], y=rb["price"], name="BTC Price", line=dict(width=2)))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Price USD", type="log"),
        legend=dict(orientation="h"),
    )
    plotly_show(fig)

with right2:
    st.subheader(f"IV 趋势（{tf}）" + ("（Deribit）" if not iv_is_proxy else "（Proxy: RV30）"))
    fig = go.Figure()
    # align price range to selected timeframe
    fig.add_trace(go.Scatter(x=hist_slice["date"], y=hist_slice["price"], name="BTC Price", yaxis="y2"))
    fig.add_trace(go.Scatter(x=iv_slice["date"], y=iv_slice["iv"] * 100, name="IV(%)"))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="IV %"),
        yaxis2=dict(title="Price USD", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    plotly_show(fig)

st.caption("说明：为避免 Cloud 环境限流，价格/历史主要用 CryptoCompare；Deribit IV 若不可用则自动回退到 RV30 代理。")
