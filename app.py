import time
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="BTC Market Dashboard", layout="wide")

# =========================================================
# Compatibility helpers (avoid Streamlit deprecation warnings)
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
def get_deribit_vol_index(days: int = 180):
    """
    Deribit volatility index data (BTC)
    Return (df, source, http_code)

    If endpoint is unavailable, raise and caller will fallback to RV30.
    """
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
            if "timestamp" in df.columns and "value" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

                # Deribit values often look like 40~120 (percent); normalize to decimal
                v = df["value"].astype(float)
                if v.median() > 2.0:
                    df["iv"] = v / 100.0
                else:
                    df["iv"] = v

                df = df[["date", "iv"]].sort_values("date").reset_index(drop=True)
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

def window_percentile(series: pd.Series, value: float, window_days: int):
    """
    percentile of value within last `window_days` points of series (assume daily-ish)
    """
    s = series.dropna().astype(float)
    if len(s) == 0 or np.isnan(value):
        return np.nan
    s2 = s.iloc[-window_days:] if len(s) >= window_days else s
    return percentile_rank(s2, value)

def score_from_metrics(vol_pct, fng_value, band_pos):
    score = 50.0
    if vol_pct is not None and not np.isnan(vol_pct):
        score += (50.0 - vol_pct) * 0.4
    if fng_value is not None and not np.isnan(fng_value):
        score += (25.0 - abs(fng_value - 50.0)) * 0.4
    if band_pos is not None and not np.isnan(band_pos):
        score += (0.25 - abs(band_pos - 0.5)) * 40.0
    return float(np.clip(score, 0, 100))

def build_rainbow_bands(df_price_all: pd.DataFrame):
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

def band_index_from_pos(pos: float):
    """
    Convert 0..1 position into Band 1..8
    """
    if np.isnan(pos):
        return None
    idx = int(np.floor(pos * 8)) + 1
    return int(np.clip(idx, 1, 8))

def band_label_from_idx(idx: int):
    labels = {
        1: "Band 1（偏低）",
        2: "Band 2（偏低）",
        3: "Band 3（中低）",
        4: "Band 4（中性）",
        5: "Band 5（中性偏高）",
        6: "Band 6（偏高）",
        7: "Band 7（偏高）",
        8: "Band 8（极高）",
    }
    return labels.get(idx, "N/A")

def current_band_position(rainbow_df: pd.DataFrame):
    last = rainbow_df.iloc[-1]
    low = last["b-2.0"]
    high = last["b+2.0"]
    p = last["price"]
    if high <= low:
        return np.nan, "N/A", None
    pos = float((p - low) / (high - low))
    idx = band_index_from_pos(pos)
    label = band_label_from_idx(idx) if idx else "N/A"
    return pos, label, idx

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

def kpi_card_like_ref(title, main_value, subline, p30=None, p90=None, extra_right=None):
    """
    Reference-like KPI card:
    - title
    - main value
    - subline
    - 30d/90d percentile bars
    """
    def bar(label, pct):
        if pct is None or np.isnan(pct):
            return ""
        pct = float(np.clip(pct, 0, 100))
        return f"""
        <div style="display:flex;align-items:center;gap:8px;margin-top:8px;">
          <div style="width:44px;font-size:11px;color:#9aa4b2;">{label}</div>
          <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:999px;height:8px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:8px;background:rgba(255,255,255,0.35);"></div>
          </div>
          <div style="width:44px;text-align:right;font-size:11px;color:#9aa4b2;">{pct:.0f}%</div>
        </div>
        """

    extra = f"<div style='font-size:11px;color:#9aa4b2;margin-top:8px;'>{extra_right}</div>" if extra_right else ""

    st.markdown(
        f"""
        <div style="padding:14px;border:1px solid rgba(255,255,255,0.08);
        border-radius:12px;background:rgba(255,255,255,0.02);">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div style="font-size:12px;color:#9aa4b2;">{title}</div>
          </div>
          <div style="font-size:22px;font-weight:700;margin-top:6px;">{main_value}</div>
          <div style="font-size:12px;color:#9aa4b2;margin-top:6px;">{subline}</div>
          {bar("30d", p30)}
          {bar("90d", p90)}
          {extra}
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# UI
# =========================================================
st.title("BTC 市场分析 Dashboard（方向1：KPI还原参考图）")

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

    hist, hist_src, _ = get_btc_history_daily(days=max(365 * 3, days))
    source_status.append(("历史日线", hist_src, "OK"))

    fng, fng_src, _ = get_fear_greed(limit=max(200, days + 30))
    source_status.append(("恐惧贪婪", fng_src, "OK"))

    try:
        iv_df, iv_src, _ = get_deribit_vol_index(days=max(180, days))
        source_status.append(("期权IV指数", iv_src, "OK"))
        iv_is_proxy = False
    except Exception:
        iv_df = None
        iv_src = "Proxy: RV30"
        source_status.append(("期权IV指数", iv_src, "FALLBACK"))
        iv_is_proxy = True

except Exception as e:
    st.error(f"数据拉取失败：{e}")
    st.stop()

with st.expander("数据源状态（点开查看）", expanded=False):
    df_status = pd.DataFrame(source_status, columns=["模块", "数据源", "状态"])
    dataframe_show(df_status)

# =========================================================
# Slice + metrics
# =========================================================
hist = hist.sort_values("date").reset_index(drop=True)
hist_slice = hist[hist["date"] >= (hist["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)

fng = fng.sort_values("date").reset_index(drop=True)
fng_slice = fng[fng["date"] >= (fng["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)

rv30 = realized_vol(hist, 30)
rv90 = realized_vol(hist, 90)

# Vol (use RV90 as DVOL proxy)
rv90_last = float(rv90.dropna().iloc[-1]["rv90"]) if rv90["rv90"].notna().any() else np.nan
rv90_p30 = window_percentile(rv90["rv90"], rv90_last, 30)
rv90_p90 = window_percentile(rv90["rv90"], rv90_last, 90)

# FNG
fng_last = float(fng_slice.iloc[-1]["value"]) if len(fng_slice) else np.nan
fng_class = fng_slice.iloc[-1]["value_classification"] if len(fng_slice) else "N/A"
fng_p30 = window_percentile(fng["value"], fng_last, 30)
fng_p90 = window_percentile(fng["value"], fng_last, 90)

# Rainbow band
rainbow = build_rainbow_bands(hist)
band_pos, band_label, band_idx = current_band_position(rainbow)
band_text = f"Band {band_idx}/8" if band_idx else "Band N/A"

# IV (Deribit or proxy)
if iv_df is None:
    iv_series = rv30.rename(columns={"rv30": "iv"}).copy().dropna().reset_index(drop=True)
else:
    iv_series = iv_df.copy().dropna().reset_index(drop=True)

iv_slice = iv_series[iv_series["date"] >= (iv_series["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)
iv_last = float(iv_slice.iloc[-1]["iv"]) if len(iv_slice) else np.nan
iv_prev = float(iv_slice.iloc[-2]["iv"]) if len(iv_slice) >= 2 else np.nan
iv_change = (iv_last - iv_prev) if (not np.isnan(iv_last) and not np.isnan(iv_prev)) else np.nan
iv_p30 = window_percentile(iv_series["iv"], iv_last, 30)
iv_p90 = window_percentile(iv_series["iv"], iv_last, 90)

# Score
vol_pct_for_score = percentile_rank(rv90["rv90"], rv90_last)
score = score_from_metrics(vol_pct_for_score, fng_last, band_pos)

# =========================================================
# KPI row (reference-like)
# =========================================================
c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])

with c1:
    plotly_show(gauge(score, f"{int(round(score))}", "较好的状态" if score >= 60 else "偏谨慎"))
    st.caption(f"BTC Spot: ${spot:,.0f}  ·  Source: {spot_src}")

with c2:
    kpi_card_like_ref(
        "IV 百分位（DVOL）",
        f"DVOL {rv90_last*100:.1f}%" if not np.isnan(rv90_last) else "N/A",
        "波动率越高，风险越高",
        p30=rv90_p30,
        p90=rv90_p90,
    )

with c3:
    kpi_card_like_ref(
        "恐惧贪婪指数",
        f"{int(round(fng_last))}（{fng_class}）" if not np.isnan(fng_last) else "N/A",
        "情绪极端 = 拥挤交易",
        p30=fng_p30,
        p90=fng_p90,
    )

with c4:
    kpi_card_like_ref(
        "比特币彩虹图",
        f"{band_label}",
        band_text,
        p30=(band_pos * 100 if not np.isnan(band_pos) else None),
        p90=None,
        extra_right="（估值带位置）"
    )

with c5:
    arrow = "▲" if (not np.isnan(iv_change) and iv_change >= 0) else "▼"
    title = "IV 趋势（Deribit）" if not iv_is_proxy else "IV 趋势（Proxy）"
    kpi_card_like_ref(
        title,
        f"{iv_last*100:.1f}%" if not np.isnan(iv_last) else "N/A",
        f"24h: {arrow} {iv_change*100:+.2f}%" if not np.isnan(iv_change) else "24h: N/A",
        p30=iv_p30,
        p90=iv_p90,
    )

st.divider()

# =========================================================
# 4-panel charts (same structure)
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
    rb = rainbow.copy()
    rb = rb[rb["date"] >= (rb["date"].max() - pd.Timedelta(days=365*3))].reset_index(drop=True)

    fig = go.Figure()
    band_keys = ["b-2.0", "b-1.5", "b-1.0", "b-0.5", "b+0.0", "b+0.5", "b+1.0", "b+1.5", "b+2.0"]
    fig.add_trace(go.Scatter(x=rb["date"], y=rb[band_keys[0]], name="Band low", line=dict(width=1), opacity=0.2))
    for i in range(1, len(band_keys)):
        fig.add_trace(go.Scatter(
            x=rb["date"], y=rb[band_keys[i]],
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

st.caption("说明：方向1已完成：顶部 KPI 采用“主值 + 30d/90d 分位条 + Band X/8”格式，更贴近参考图。")
