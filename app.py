import time
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="BTC Wyckoff Dashboard", layout="wide")

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
# Safe requests + retries
# =========================================================
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BTC-Wyckoff-Dashboard/1.0; +https://streamlit.app)",
    "Accept": "application/json",
}

def safe_get(url, params=None, timeout=15, retries=3):
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
# Data sources (avoid CoinGecko)
# =========================================================
@st.cache_data(ttl=30)
def get_btc_spot_usd():
    data, code, _ = safe_get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=12, retries=3)
    if data and "data" in data and "amount" in data["data"]:
        return float(data["data"]["amount"]), "Coinbase", 200
    coinbase_code = code

    data, code, _ = safe_get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"}, timeout=12, retries=3)
    if data and "price" in data:
        return float(data["price"]), "Binance", 200
    binance_code = code

    data, code, _ = safe_get("https://min-api.cryptocompare.com/data/price", params={"fsym": "BTC", "tsyms": "USD"}, timeout=12, retries=3)
    if data and "USD" in data:
        return float(data["USD"]), "CryptoCompare", 200
    cc_code = code

    raise RuntimeError(f"现价数据源全部失败：Coinbase HTTP={coinbase_code}, Binance HTTP={binance_code}, CryptoCompare HTTP={cc_code}")

@st.cache_data(ttl=600)
def get_btc_history_daily(days: int = 365):
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
        return df[["date", "price"]].sort_values("date").reset_index(drop=True), "CryptoCompare", 200
    raise RuntimeError(f"历史价格失败（CryptoCompare）。HTTP={code} err={err}")

@st.cache_data(ttl=600)
def get_fear_greed(limit=200):
    data, code, err = safe_get("https://api.alternative.me/fng/", params={"limit": limit, "format": "json"}, timeout=20, retries=3)
    if data and "data" in data:
        df = pd.DataFrame(data["data"])
        df["value"] = df["value"].astype(float)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df = df.sort_values("date").reset_index(drop=True)[["date", "value", "value_classification"]]
        return df, "alternative.me", 200
    raise RuntimeError(f"恐惧贪婪指数失败（alternative.me）。HTTP={code} err={err}")

@st.cache_data(ttl=120)
def get_btc_history_hourly(hours: int = 24 * 60):
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
# Utils & indicators
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

def zscore_last(series: pd.Series, window: int = 120):
    s = series.dropna()
    if len(s) < 20:
        return np.nan
    s2 = s.iloc[-window:] if len(s) > window else s
    mu = float(s2.mean())
    sd = float(s2.std(ddof=0)) if float(s2.std(ddof=0)) > 1e-9 else 1e-9
    return float((s2.iloc[-1] - mu) / sd)

def swing_highs(df: pd.DataFrame, left: int = 2, right: int = 2):
    highs = df["high"].values
    idxs = []
    for i in range(left, len(df) - right):
        if highs[i] > max(highs[i-left:i]) and highs[i] > max(highs[i+1:i+1+right]):
            idxs.append(i)
    return idxs

def swing_lows(df: pd.DataFrame, left: int = 2, right: int = 2):
    lows = df["low"].values
    idxs = []
    for i in range(left, len(df) - right):
        if lows[i] < min(lows[i-left:i]) and lows[i] < min(lows[i+1:i+1+right]):
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

    body, rng, upper, _ = candle_features(cur)
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
    if df_4h is None or len(df_4h) < 30:
        return False, ["数据不足"]
    last_close = float(df_4h.iloc[-1]["close"])
    near_top = last_close >= box_high * (1 - near_pct)
    lh_ok, lh_info = detect_lower_high(df_4h, lookback_swings=4)
    pat_ok, pat_info = detect_bearish_patterns(df_4h)
    reasons = [
        (f"接近上沿阈值：≥ {box_high*(1-near_pct):.0f}（当前 close={last_close:.0f}）" if near_top else f"未接近上沿（当前 close={last_close:.0f}）"),
        lh_info,
        pat_info,
        f"模式：{mode}",
    ]
    if mode == "严格":
        triggered = near_top and lh_ok and pat_ok
    else:
        triggered = near_top and (lh_ok or pat_ok)
    return triggered, reasons

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

# ---- SC / ST detectors (long side)
def detect_sc_event(df_1h: pd.DataFrame, sc_low: float, sc_high: float,
                    drop_24h: float = -0.06, range_24h: float = 0.06, vol_z: float = 1.5):
    """
    SC proxy:
    - price enters SC zone (40k-49k by default)
    - 24h return <= drop_24h (e.g. -6%)
    - 24h range >= range_24h (big volatility)
    - volume zscore >= vol_z (spike)
    """
    if len(df_1h) < 48:
        return False, "数据不足"

    d = df_1h.tail(48).copy()  # last 48 hours
    last = float(d["close"].iloc[-1])
    in_zone = (last <= sc_low) and (last >= sc_high) if sc_low >= sc_high else (last >= sc_low and last <= sc_high)

    ret = (float(d["close"].iloc[-1]) / float(d["close"].iloc[0]) - 1.0)
    rng = (float(d["high"].max()) / float(d["low"].min()) - 1.0)
    vz = zscore_last(df_1h["volume"], window=240)  # ~10 days window

    ok = in_zone and (ret <= drop_24h) and (rng >= range_24h) and (not np.isnan(vz) and vz >= vol_z)
    info = f"区间:{'是' if in_zone else '否'}  24h跌幅:{ret*100:+.1f}%  24h振幅:{rng*100:.1f}%  量能Z:{vz:.2f}"
    return ok, info

def detect_st_event(df_4h: pd.DataFrame, lookback: int = 120, new_low_tol: float = 0.01, vol_shrink: float = 0.8):
    """
    ST proxy (retest):
    - recent low exists
    - a later pullback tests near that low but doesn't make meaningfully new low (>= -tol)
    - volume on retest is lower than volume during breakdown/impulse (shrink)
    - and an HL starts to form: last swing low higher than previous swing low (simple)
    """
    if len(df_4h) < 60:
        return False, "数据不足"

    df = df_4h.tail(lookback).copy()

    # Identify swing lows
    lows_idx = swing_lows(df, left=2, right=2)
    if len(lows_idx) < 3:
        return False, "摆动低点不足"

    # take last 3 swing lows: L1, L2, L3 (time order)
    i1, i2, i3 = lows_idx[-3], lows_idx[-2], lows_idx[-1]
    L1 = float(df.iloc[i1]["low"])
    L2 = float(df.iloc[i2]["low"])
    L3 = float(df.iloc[i3]["low"])

    # "not new low": L3 >= L2*(1 - new_low_tol)
    not_new_low = (L3 >= L2 * (1 - new_low_tol))

    # HL structure proxy: L3 >= L2 (strict HL) OR at least "small new low" already captured by tol
    hlish = (L3 >= L2)

    # volume shrink on retest: compare volume near L3 vs volume near L2
    v2 = float(df.iloc[max(i2-1,0):min(i2+2,len(df))]["volume"].mean())
    v3 = float(df.iloc[max(i3-1,0):min(i3+2,len(df))]["volume"].mean())
    vol_ok = (v3 <= v2 * vol_shrink)

    ok = not_new_low and vol_ok and (hlish or not_new_low)
    info = f"L2:{L2:.0f}  L3:{L3:.0f}  不创新低:{'是' if not_new_low else '否'}  量缩:{'是' if vol_ok else '否'}"
    return ok, info

# =========================================================
# UI elements
# =========================================================
def badge(text, ok=True):
    color = "#2ecc71" if ok else "#f39c12"
    st.markdown(
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.10);"
        f"color:{color};font-size:12px;margin-right:6px;margin-bottom:6px;'>{text}</span>",
        unsafe_allow_html=True
    )

def push_event(events, name, ok, detail):
    if ok:
        events.append({
            "time": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "event": name,
            "detail": detail
        })

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

def score_from_metrics(vol_pct, fng_value):
    score = 50.0
    if vol_pct is not None and not np.isnan(vol_pct):
        score += (50.0 - vol_pct) * 0.4
    if fng_value is not None and not np.isnan(fng_value):
        score += (25.0 - abs(fng_value - 50.0)) * 0.4
    return float(np.clip(score, 0, 100))

# =========================================================
# UI
# =========================================================
st.title("BTC 路线图总览：箱体→破位→SC→ST（二测）")

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

# Sidebar: parameters
with st.sidebar:
    st.markdown("## 核心区间参数（可改）")
    # A/B: box
    upper_zone_lo = st.number_input("箱体上沿加空区下限", value=70000, step=500)
    upper_zone_hi = st.number_input("箱体上沿加空区上限", value=72000, step=500)
    upper_level = st.number_input("箱体上沿关键位（UT/顶部）", value=72000, step=500)
    lower_level = st.number_input("箱体下沿关键位（破位回踩）", value=60000, step=500)
    retest_tol = st.slider("回踩容差（%）", min_value=0.2, max_value=2.0, value=0.6, step=0.1) / 100.0
    ut_lookback = st.slider("UT 收回窗口（4H根数）", min_value=2, max_value=12, value=6, step=1)
    near_top_pct = st.slider("顶部判定：接近上沿阈值（%）", min_value=0.5, max_value=5.0, value=1.5, step=0.1) / 100.0
    top_mode = st.radio("顶部判定器模式", ["宽松", "严格"], index=0)

    st.markdown("---")
    st.markdown("## SC / ST 做多区参数（可改）")
    sc_zone_high = st.number_input("SC 区间上沿（高）", value=49000, step=500)
    sc_zone_low = st.number_input("SC 区间下沿（低）", value=40000, step=500)

    sc_drop = st.slider("SC：近24h跌幅阈值（%）", min_value=2.0, max_value=20.0, value=6.0, step=0.5) / 100.0
    sc_range = st.slider("SC：近24h振幅阈值（%）", min_value=2.0, max_value=25.0, value=6.0, step=0.5) / 100.0
    sc_volz = st.slider("SC：量能激增Z阈值", min_value=0.5, max_value=4.0, value=1.5, step=0.1)

    st.markdown("---")
    st.markdown("## ST（二测）条件（可改）")
    st_new_low_tol = st.slider("ST：允许的新低幅度（%）", min_value=0.0, max_value=5.0, value=1.0, step=0.2) / 100.0
    st_vol_shrink = st.slider("ST：回测量缩比例（≤）", min_value=0.3, max_value=1.2, value=0.8, step=0.05)

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
    source_status.append(("小时线（策略/SC）", "CryptoCompare", "OK"))

except Exception as e:
    st.error(f"数据拉取失败：{e}")
    st.stop()

with st.expander("数据源状态（点开查看）", expanded=False):
    dataframe_show(pd.DataFrame(source_status, columns=["模块", "数据源", "状态"]))

# Prepare data
hist = hist.sort_values("date").reset_index(drop=True)
hist_slice = hist[hist["date"] >= (hist["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)

fng = fng.sort_values("date").reset_index(drop=True)
fng_slice = fng[fng["date"] >= (fng["date"].max() - pd.Timedelta(days=days))].reset_index(drop=True)

df_4h = resample_ohlcv(df_1h, "4H")
df_8h = resample_ohlcv(df_1h, "8H")

# KPIs
rv90 = realized_vol(hist, 90)
rv90_last = float(rv90.dropna().iloc[-1]["rv90"]) if rv90["rv90"].notna().any() else np.nan
vol_pct = percentile_rank(rv90["rv90"], rv90_last)

fng_last = float(fng_slice.iloc[-1]["value"]) if len(fng_slice) else np.nan
score = score_from_metrics(vol_pct, fng_last)

# =========================================================
# Event feed (trigger list)
# =========================================================
events = []

# Short-side signals
ut_triggered, ut_info = detect_ut_fake_breakout(df_4h, float(upper_level), lookback_bars=int(ut_lookback))
br_triggered, br_info = detect_break_retest_fail(df_4h, float(lower_level), tolerance=float(retest_tol))
top_triggered, top_reasons = top_detector(df_4h, float(upper_level), near_pct=float(near_top_pct), mode=str(top_mode))

push_event(events, "UT 假突破（更优空点）", ut_triggered, ut_info)
push_event(events, "破位回踩失败（确认空点）", br_triggered, br_info)
push_event(events, f"顶部判定器触发（{top_mode}）", top_triggered, " / ".join(top_reasons[:3]))

# Long-side signals
# Note: sc_zone_high > sc_zone_low typical; handle in detector
sc_triggered, sc_info = detect_sc_event(
    df_1h,
    sc_low=float(sc_zone_high),
    sc_high=float(sc_zone_low),
    drop_24h=-float(sc_drop),
    range_24h=float(sc_range),
    vol_z=float(sc_volz)
)
push_event(events, "SC 冲击信号（试探多区）", sc_triggered, sc_info)

st_triggered, st_info = detect_st_event(
    df_4h,
    lookback=180,
    new_low_tol=float(st_new_low_tol),
    vol_shrink=float(st_vol_shrink)
)
push_event(events, "ST 二测信号（主仓做多区）", st_triggered, st_info)

# Render event feed (latest first)
st.subheader("触发事件提示（打开就知道发生了什么）")
if len(events) == 0:
    st.info("暂无触发事件：当前更偏向‘等位置/等信号’。")
else:
    df_evt = pd.DataFrame(events)[["time", "event", "detail"]]
    dataframe_show(df_evt)

st.divider()

# =========================================================
# Top KPIs summary
# =========================================================
c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

with c1:
    plotly_show(gauge(score, f"{int(round(score))}", "市场状态（量化代理）"))
    st.caption(f"BTC Spot: ${spot:,.0f}  ·  Source: {spot_src}")

with c2:
    st.metric("箱体上沿（做空关注）", f"{upper_zone_lo:,.0f}–{upper_zone_hi:,.0f}")
with c3:
    st.metric("箱体下沿（破位确认）", f"{lower_level:,.0f}")
with c4:
    st.metric("SC 做多等待区", f"{sc_zone_low:,.0f}–{sc_zone_high:,.0f}")

st.divider()

# =========================================================
# Strategy Panels (Short & Long)
# =========================================================
current_price = float(spot)

# ---- A/B: short accumulation map
st.subheader("做空累计（A/B 段）：上沿做空 / UT / 破位回踩失败")

in_upper_zone = in_range(current_price, float(upper_zone_lo), float(upper_zone_hi))
in_mid_box = in_range(current_price, float(lower_level), float(upper_zone_lo))
below_lower = current_price < float(lower_level)

colA, colB, colC = st.columns([1.2, 1.5, 1.8])

with colA:
    st.markdown("**位置层**")
    badge(f"上沿加空区 {upper_zone_lo:,.0f}–{upper_zone_hi:,.0f}", in_upper_zone)
    badge("箱体中部（不加仓）", in_mid_box)
    badge(f"跌破下沿 < {lower_level:,.0f}", below_lower)

with colB:
    st.markdown("**结构层**")
    badge(f"UT 假突破：{ut_info}", ut_triggered)
    badge(f"破位回踩失败：{br_info}", br_triggered)
    badge(f"顶部判定器（{top_mode}）", top_triggered)
    with st.expander("顶部判定器细节", expanded=False):
        for r in top_reasons:
            st.write(f"- {r}")

with colC:
    st.markdown("**操作层（空单）**")
    if br_triggered:
        st.info("✅ **确认空点**：破位后回踩失败（更高胜率的加空方式）。")
    elif ut_triggered:
        st.info("✅ **更优空点**：UT 假突破（诱多后收回）。")
    elif in_upper_zone and top_triggered:
        st.warning("🟡 **上沿 + 顶部确认**：可考虑分批加空（符合“位置好 + 短周期到顶”）。")
    elif in_upper_zone:
        st.success("🟢 进入上沿关注区，但缺顶部确认：更适合等触发再动。")
    else:
        st.success("🟢 不在理想加仓区：按纪律更像‘等位置/等信号’。")

st.divider()

# ---- C/D: long roadmap (SC -> ST)
st.subheader("做多路线（C/D 段）：SC 试探多 → ST（二测）主仓")

# SC zone check
sc_in_zone = (current_price <= float(sc_zone_high)) and (current_price >= float(sc_zone_low))
colL1, colL2, colL3 = st.columns([1.2, 1.5, 1.8])

with colL1:
    st.markdown("**位置层（做多）**")
    badge(f"SC 等待区 {sc_zone_low:,.0f}–{sc_zone_high:,.0f}", sc_in_zone)
    badge("未进入 SC 区（继续等）", not sc_in_zone)

with colL2:
    st.markdown("**SC 冲击信号（试探仓）**")
    badge(f"SC 信号：{sc_info}", sc_triggered)

    st.markdown("**ST 二测信号（主仓）**")
    badge(f"ST 信号：{st_info}", st_triggered)

with colL3:
    st.markdown("**操作层（多单）**")
    if st_triggered:
        st.info("✅ **主仓做多区（ST 二测）**：不创新低 + 量缩 + 结构开始抬高（按规则代理）。")
    elif sc_triggered:
        st.warning("🟡 **试探多区（SC 冲击）**：只适合小仓试探，主仓等待 ST（二测）确认。")
    elif sc_in_zone:
        st.success("🟢 价格进入 SC 区，但尚未触发“冲击信号”：耐心等‘更恐慌的一脚’或结构确认。")
    else:
        st.success("🟢 未到做多等待区：按路线图继续等 SC → ST，而不是提前抄底。")

st.caption("注：SC/ST 为规则化代理信号（跌幅/振幅/量能 & 不创新低/量缩/摆动结构），用于把你的文字框架变成可执行提示。")

st.divider()

# =========================================================
# Charts
# =========================================================
left1, right1 = st.columns(2)
left2, right2 = st.columns(2)

with left1:
    st.subheader("价格（日线）+ 关键位")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_slice["date"], y=hist_slice["price"], name="BTC Price"))
    fig.add_hline(y=float(upper_level), line_width=1, opacity=0.5)
    fig.add_hline(y=float(lower_level), line_width=1, opacity=0.5)
    fig.add_hrect(y0=float(sc_zone_low), y1=float(sc_zone_high), opacity=0.08, line_width=0)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(title="Price USD"))
    plotly_show(fig)

with right1:
    st.subheader("4H K线（近60天）")
    d = df_4h.tail(300).copy()
    fig = go.Figure(data=[go.Candlestick(
        x=d["date"], open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="4H"
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
    st.subheader("小时线（SC 冲击观察：近7天）")
    h = df_1h.tail(24 * 7).copy()
    fig = go.Figure(data=[go.Candlestick(
        x=h["date"], open=h["open"], high=h["high"], low=h["low"], close=h["close"], name="1H"
    )])
    fig.add_hrect(y0=float(sc_zone_low), y1=float(sc_zone_high), opacity=0.08, line_width=0)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    plotly_show(fig)

st.caption("执行清单：只在上沿+顶部确认加空；UT 更优；60k 破位后回踩失败确认空；SC 区仅试探多；主仓等 ST（二测）确认。")
