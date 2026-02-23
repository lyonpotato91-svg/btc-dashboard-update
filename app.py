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
    - Never raise_for_status
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

@st.cache_data(ttl=120)
def get_btc_history_hourly(hours: int = 24 * 60):
    """
    CryptoCompare histohour
    Return df: date, open, high, low, close, volume
    """
    limit = int(min(max(hours, 24), 2000)) - 1
    data, code, err = safe_get(
        "https://min-api.cryptocompare.com/data/v2/histohour",
        params={"fsym": "BTC", "tsym": "USD", "limit": limit},
        timeout=20,
        retries=3,
    )
    if data and data.get("Response") == "Success":
        rows = data["Data"]["Data"]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"volumeto": "volume"})
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)

    raise RuntimeError(f"小时线失败（CryptoCompare）。HTTP={code} err={err}")

# =========================================================
# Resample & Wyckoff-ish top detector (LH + candlesticks)
# =========================================================
def resample_ohlcv(df_1h: pd.DataFrame, rule: str):
    d = df_1h.set_index("date").copy()
    out = d.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()
    return out

def in_range(x, lo, hi):
    return (x >= lo) and (x <= hi)

def detect_ut_fake_breakout(df_tf: pd.DataFrame, level: float, lookback_bars: int = 6):
    if len(df_tf) < lookback_bars + 3:
        return False, "数据不足"
    closes = df_tf["close"].values
    idxs = np.where(closes > level)[0]
    if len(idxs) == 0:
        return False, "未突破上沿"
    last_break = idxs[-1]
    end = min(len(df_tf) - 1, last_break + lookback_bars)
    after = df_tf.iloc[last_break:end + 1]
    if (after["close"] < level).any():
        return True, f"突破后 {lookback_bars} 根内收回下方"
    return False, "突破但未收回（观察）"

def detect_break_retest_fail(df_tf: pd.DataFrame, level: float, tolerance: float = 0.006, lookback: int = 60):
    if len(df_tf) < 30:
        return False, "数据不足"
    df = df_tf.iloc[-lookback:].copy() if len(df_tf) > lookback else df_tf.copy()
    idxs = np.where(df["close"].values < level)[0]
    if len(idxs) == 0:
        return False, "未跌破下沿"
    last_break = idxs[-1]
    after = df.iloc[last_break:].copy()
    lo, hi = level * (1 - tolerance), level * (1 + tolerance)
    cond = (after["high"].between(lo, hi)) & (after["close"] < level)
    if cond.any():
        return True, f"回踩触及 {lo:.0f}-{hi:.0f} 但收不回 {level:.0f}"
    return False, "跌破后尚未出现回踩失败"

def swing_highs(df: pd.DataFrame, left: int = 2, right: int = 2):
    highs = df["high"].values
    idxs = []
    for i in range(left, len(df) - right):
        if highs[i] > max(highs[i-left:i]) and highs[i] > max(highs[i+1:i+1+right]):
            idxs.append(i)
    return idxs

def detect_lower_high(df_tf: pd.DataFrame, lookback_swings: int = 4):
    if len(df_tf) < 20:
        return False, "数据不足"
    idxs = swing_highs(df_tf, left=2, right=2)
    if len(idxs) < 2:
        return False, "未形成足够摆动高点"
    idxs = idxs[-lookback_swings:] if len(idxs) > lookback_swings else idxs
    if len(idxs) < 2:
        return False, "摆动点不足"
    last_i, prev_i = idxs[-1], idxs[-2]
    last_high = float(df_tf.iloc[last_i]["high"])
    prev_high = float(df_tf.iloc[prev_i]["high"])
    if last_high < prev_high:
        return True, f"LH：{last_high:.0f} < {prev_high:.0f}"
    return False, f"非LH：{last_high:.0f} ≥ {prev_high:.0f}"

def candle_features(row):
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return body, rng, upper, lower

def detect_bearish_patterns(df_tf: pd.DataFrame):
    if len(df_tf) < 3:
        return False, "数据不足"
    cur = df_tf.iloc[-1]
    prev = df_tf.iloc[-2]

    body, rng, upper, lower = candle_features(cur)

    close_lower_half = float(cur["close"]) <= (float(cur["low"]) + 0.5 * (float(cur["high"]) - float(cur["low"])))
    shoot = (upper / rng >= 0.55) and (body / rng <= 0.30) and close_lower_half

    prev_o, prev_c = float(prev["open"]), float(prev["close"])
    cur_o, cur_c = float(cur["open"]), float(cur["close"])
    prev_low_body = min(prev_o, prev_c)
    prev_high_body = max(prev_o, prev_c)
    cur_low_body = min(cur_o, cur_c)
    cur_high_body = max(cur_o, cur_c)
    bearish = (cur_c < cur_o) and (cur_low_body <= prev_low_body) and (cur_high_body >= prev_high_body)

    if shoot and bearish:
        return True, "长上影 + 看跌吞没（强顶部信号）"
    if shoot:
        return True, "长上影（Upthrust/Shooting star）"
    if bearish:
        return True, "看跌吞没（Bearish engulfing）"
    return False, "未出现典型顶部K线"

def top_detector(df_4h: pd.DataFrame, box_high: float, near_pct: float = 0.015, mode: str = "宽松"):
    """
    mode:
      - "宽松": near_top AND (LH OR bearish_pattern)
      - "严格": near_top AND (LH AND bearish_pattern)
    Return (triggered, reasons)
    """
    if df_4h is None or len(df_4h) < 30:
        return False, ["数据不足"]

    last_close = float(df_4h.iloc[-1]["close"])
    near_top = last_close >= box_high * (1 - near_pct)

    lh_ok, lh_info = detect_lower_high(df_4h, lookback_swings=4)
    pat_ok, pat_info = detect_bearish_patterns(df_4h)

    reasons = []
    reasons.append(f"接近上沿阈值：≥ {box_high*(1-near_pct):.0f}（当前 close={last_close:.0f}）" if near_top else f"未接近上沿（当前 close={last_close:.0f}）")
    reasons.append(lh_info)
    reasons.append(pat_info)
    reasons.append(f"模式：{mode}")

    if mode == "严格":
        triggered = near_top and lh_ok and pat_ok
    else:
        triggered = near_top and (lh_ok or pat_ok)

    return triggered, reasons

# =========================================================
# Optional KPIs
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

def current_band_position(rainbow_df: pd.DataFrame):
    last = rainbow_df.iloc[-1]
    low = last["b-2.0"]
    high = last["b+2.0"]
    p = last["price"]
    if high <= low:
        return np.nan, "N/A"
    pos = float((p - low) / (high - low))
    idx = int(np.clip(int(np.floor(pos * 8)) + 1, 1, 8))
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
    return pos, labels.get(idx, "N/A")

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

# =========================================================
# UI
# =========================================================
st.title("BTC 市场分析 Dashboard（含威科夫做空累计提示 + 顶部判定器 宽松/严格）")

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

with st.sidebar:
    st.markdown("## 策略参数（可改）")
    upper_zone_lo = st.number_input("箱体上沿加空区下限", value=70000, step=500)
    upper_zone_hi = st.number_input("箱体上沿加空区上限", value=72000, step=500)
    upper_level = st.number_input("箱体上沿关键位（UT判断）", value=72000, step=500)
    lower_level = st.number_input("箱体下沿关键位（破位回踩）", value=60000, step=500)
    retest_tol = st.slider("回踩容差（%）", min_value=0.2, max_value=2.0, value=0.6, step=0.1) / 100.0
    ut_lookback = st.slider("UT 收回窗口（4H根数）", min_value=2, max_value=12, value=6, step=1)
    near_top_pct = st.slider("顶部判定：接近上沿阈值（%）", min_value=0.5, max_value=5.0, value=1.5, step=0.1) / 100.0

    top_mode = st.radio("顶部判定器模式", ["宽松", "严格"], index=0)
    st.caption("宽松：接近上沿 AND（LH 或 顶部K线）\n\n严格：接近上沿 AND（LH 且 顶部K线）")

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

    df_1h = get_btc_history_hourly(hours=24 * 60)
    source_status.append(("小时线(用于4H/8H)", "CryptoCompare", "OK"))

except Exception as e:
    st.error(f"数据拉取失败：{e}")
    st.stop()

with st.expander("数据源状态（点开查看）", expanded=False):
    dataframe_show(pd.DataFrame(source_status, columns=["模块", "数据源", "状态"]))

# Prepare datasets
hist = hist.sort_values("date").reset_index(drop=True)
hist_slice = hist[hist["date"] >= (hist["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)

fng = fng.sort_values("date").reset_index(drop=True)
fng_slice = fng[fng["date"] >= (fng["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)

df_4h = resample_ohlcv(df_1h, "4H")
df_8h = resample_ohlcv(df_1h, "8H")

# KPIs (optional)
rv90 = realized_vol(hist, 90)
rv90_last = float(rv90.dropna().iloc[-1]["rv90"]) if rv90["rv90"].notna().any() else np.nan
vol_pct = percentile_rank(rv90["rv90"], rv90_last)

fng_last = float(fng_slice.iloc[-1]["value"]) if len(fng_slice) else np.nan

rainbow = build_rainbow_bands(hist)
band_pos, band_label = current_band_position(rainbow)

score = score_from_metrics(vol_pct, fng_last, band_pos)

# Top KPIs
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    plotly_show(gauge(score, f"{int(round(score))}", "综合市场状态"))
    st.caption(f"BTC Spot: ${spot:,.0f}  ·  Source: {spot_src}")
with c2:
    st.metric("箱体上沿", f"{upper_zone_lo:,.0f} – {upper_zone_hi:,.0f}", "优先加空关注区")
with c3:
    st.metric("箱体下沿", f"{lower_level:,.0f}", "破位回踩确认区")

st.divider()

# Strategy Panel
st.subheader("策略提示（做空累计：上沿做空 + UT + 破位回踩 + 顶部判定器）")
current_price = float(spot)

in_upper_zone = in_range(current_price, upper_zone_lo, upper_zone_hi)
in_mid_box = in_range(current_price, lower_level, upper_zone_lo)
below_lower = current_price < lower_level

ut_triggered, ut_info = detect_ut_fake_breakout(df_4h, upper_level, lookback_bars=int(ut_lookback))
br_triggered, br_info = detect_break_retest_fail(df_4h, lower_level, tolerance=float(retest_tol))
top_triggered, top_reasons = top_detector(df_4h, upper_level, near_pct=float(near_top_pct), mode=str(top_mode))

def badge(text, ok=True):
    color = "#2ecc71" if ok else "#f39c12"
    st.markdown(
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.10);"
        f"color:{color};font-size:12px;margin-right:6px;margin-bottom:6px;'>{text}</span>",
        unsafe_allow_html=True
    )

colA, colB, colC = st.columns([1.2, 1.5, 1.8])

with colA:
    st.markdown("**位置层（现在在哪）**")
    badge(f"上沿加空区 {upper_zone_lo:,.0f}–{upper_zone_hi:,.0f}", in_upper_zone)
    badge("箱体中部（不优先加仓）", in_mid_box)
    badge(f"跌破下沿 < {lower_level:,.0f}（等待回踩）", below_lower)

with colB:
    st.markdown("**结构层（形态/确认）**")
    badge(f"UT 假突破：{ut_info}", ut_triggered)
    badge(f"破位回踩失败：{br_info}", br_triggered)

    st.markdown("**顶部判定器（更贴威科夫）**")
    badge(f"顶部判定器触发（{top_mode}）", top_triggered)
    with st.expander("顶部判定器细节", expanded=False):
        for r in top_reasons:
            st.write(f"- {r}")

with colC:
    st.markdown("**操作层（Dashboard 提示）**")
    if br_triggered:
        st.info("✅ **确认空点：破位后回踩失败**\n\n已跌破下沿并回踩不过（结构确认转弱），按你的框架属于更“稳健”的加空类型。")
    elif ut_triggered:
        st.info("✅ **更优空点：UT 假突破**\n\n突破上沿后迅速收回区间，典型“诱多+供应回归”。可作为更优的加空触发。")
    elif in_upper_zone and top_triggered:
        st.warning("🟡 **可考虑分批加空：上沿 + 顶部判定器确认**\n\n你已经在上沿区域，同时出现 LH/顶部K线失败形态，符合“位置好 + 到顶部才加空”。")
    elif in_upper_zone:
        st.success("🟢 **进入加空关注区（上沿）**\n\n但还缺顶部确认（按模式判定）。更适合耐心等触发再动。")
    else:
        st.success("🟢 **当前不在理想加仓区**\n\n按你的框架更像‘等位置/等信号’，避免在箱体中段情绪化加仓。")

st.caption("说明：顶部判定器=接近上沿 +（LH 与/或 顶部失败K线），可切换宽松/严格以调整触发频率。")

st.divider()

# Charts
left1, right1 = st.columns(2)
left2, right2 = st.columns(2)

with left1:
    st.subheader("价格（日线）+ 关键位")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_slice["date"], y=hist_slice["price"], name="BTC Price"))
    fig.add_hline(y=float(upper_level), line_width=1, opacity=0.5)
    fig.add_hline(y=float(lower_level), line_width=1, opacity=0.5)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(title="Price USD"))
    plotly_show(fig)

with right1:
    st.subheader("4H K线（近60天）")
    d = df_4h.tail(300).copy()
    fig = go.Figure(data=[go.Candlestick(
        x=d["date"],
        open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name="4H"
    )])
    fig.add_hline(y=float(upper_level), line_width=1, opacity=0.5)
    fig.add_hline(y=float(lower_level), line_width=1, opacity=0.5)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    plotly_show(fig)

with left2:
    st.subheader("恐惧贪婪（FNG）+ 价格")
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

with right2:
    st.subheader("彩虹带（估值带）")
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

st.caption("提示：上沿等顶部确认加空；更优=UT；确认=破位回踩失败；做多等更恐慌的SC/二测。")
