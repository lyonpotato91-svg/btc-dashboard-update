
import os
import re
import time
import io
import sqlite3
import hashlib
import urllib.parse
import json
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="B站运营数据Dashboard", layout="wide")
_BOOT_NOTICE = st.empty()
_BOOT_NOTICE.info("应用正在启动，请稍等。如果长时间停在这里，通常是数据库/备份文件过大或云端正在重启。")

# =========================
# Constants
# =========================
BASELINE_PROJECT = "__BASELINE__"       # 隐藏项目：不出现在项目归档/筛选里

# ✅ DB固定到 app.py 同目录（避免工作目录变化导致“新建空库→基准全没”）
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "bili_dashboard.db")

# ✅ 自动备份（尽可能减少“每周打开空库”）
BACKUP_DIR = os.path.join(APP_DIR, "backup")
BACKUP_LATEST_CSV = os.path.join(BACKUP_DIR, "backup_latest.csv")
BACKUP_SNAPSHOTS_CSV = os.path.join(BACKUP_DIR, "backup_snapshots_latest.csv")

TABLE_NAME = "videos"
SNAPSHOT_TABLE_NAME = "video_snapshots"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Origin": "https://space.bilibili.com",
    "Referer": "https://space.bilibili.com/",
}

# =========================
# DB
# =========================
def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with db_conn() as conn:
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            project TEXT NOT NULL,
            bvid TEXT NOT NULL,
            url TEXT,
            title TEXT,
            pubdate TEXT,
            owner_mid TEXT,
            owner_name TEXT,
            view INTEGER,
            like INTEGER,
            coin INTEGER,
            favorite INTEGER,
            reply INTEGER,
            danmaku INTEGER,
            share INTEGER,
            fans_delta INTEGER,
            baseline_for TEXT,
            data_type TEXT,
            fetched_at TEXT,
            PRIMARY KEY (project, bvid)
        )
        """)
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {SNAPSHOT_TABLE_NAME} (
            project TEXT NOT NULL,
            bvid TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            fetched_at TEXT,
            pubdate TEXT,
            owner_mid TEXT,
            owner_name TEXT,
            title TEXT,
            view INTEGER,
            like INTEGER,
            coin INTEGER,
            favorite INTEGER,
            reply INTEGER,
            danmaku INTEGER,
            share INTEGER,
            engagement INTEGER,
            engagement_rate REAL,
            deep_signal_ratio REAL,
            data_type TEXT,
            PRIMARY KEY (project, bvid, snapshot_date)
        )
        """)
        conn.commit()

def _ensure_backup_dir():
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
    except Exception:
        pass

def _save_backup_csv(df_all: pd.DataFrame):
    """
    将全量库写到 backup_latest.csv
    """
    if df_all is None or df_all.empty:
        return
    _ensure_backup_dir()
    try:
        df_all.to_csv(BACKUP_LATEST_CSV, index=False, encoding="utf-8-sig")
    except Exception:
        # 备份失败不要影响主流程
        pass

def _save_snapshot_backup_csv(df_snap: pd.DataFrame):
    if df_snap is None or df_snap.empty:
        return
    _ensure_backup_dir()
    try:
        df_snap.to_csv(BACKUP_SNAPSHOTS_CSV, index=False, encoding="utf-8-sig")
    except Exception:
        pass

def _try_restore_from_backup() -> bool:
    """
    如果DB为空，尝试从 backup_latest.csv 恢复到DB
    返回是否恢复成功
    """
    if not os.path.exists(BACKUP_LATEST_CSV): return False
    try:
        if os.path.getsize(BACKUP_LATEST_CSV) > 80 * 1024 * 1024: return False
    except Exception:
        pass

    try:
        with open(BACKUP_LATEST_CSV, "rb") as f:
            raw = f.read()
    except Exception:
        return False

    df_imp = None
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            df_imp = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except Exception:
            continue

    if df_imp is None or df_imp.empty: return False

    df_imp = normalize_df(df_imp)
    if "fetched_at" not in df_imp.columns: df_imp["fetched_at"] = pd.Timestamp.now()

    df_imp["pubdate"] = pd.to_datetime(df_imp["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    df_imp["fetched_at"] = pd.to_datetime(df_imp["fetched_at"], errors="coerce").fillna(pd.Timestamp.now()).dt.strftime("%Y-%m-%d %H:%M:%S")

    upsert_rows(df_imp, skip_backup=True)
    return True

def load_all_rows() -> pd.DataFrame:
    """
    读取DB；如果为空，自动尝试从 backup_latest.csv 恢复一次再读
    """
    try:
        init_db()
        with db_conn() as conn:
            df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

        if df is not None and not df.empty:
            return df

        # DB为空 -> 尝试恢复；恢复失败也返回当前空表，不让页面白屏
        try:
            restored = _try_restore_from_backup()
        except Exception:
            restored = False
        if restored:
            with db_conn() as conn:
                df2 = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
            return df2

        return df
    except Exception as e:
        st.error("数据库读取失败，页面已进入保护模式。请检查数据库/备份文件是否损坏。")
        st.exception(e)
        return pd.DataFrame()

def upsert_rows(df_new: pd.DataFrame, skip_backup: bool = False):
    if df_new is None or df_new.empty:
        return
    init_db()
    cols = [
        "project","bvid","url","title","pubdate","owner_mid","owner_name",
        "view","like","coin","favorite","reply","danmaku","share","fans_delta",
        "baseline_for","data_type","fetched_at"
    ]
    df_new = df_new.copy()
    for c in cols:
        if c not in df_new.columns:
            df_new[c] = None
    df_new = df_new[cols]

    records = []
    for _, r in df_new.iterrows():
        records.append(tuple(None if pd.isna(v) else v for v in r.tolist()))

    placeholders = ",".join(["?"] * len(cols))
    colnames = ",".join(cols)
    sql = f"INSERT OR REPLACE INTO {TABLE_NAME} ({colnames}) VALUES ({placeholders})"
    with db_conn() as conn:
        conn.executemany(sql, records)
        conn.commit()

    if not skip_backup:
        try:
            _record_video_snapshots(df_new)
        except Exception:
            pass

    # ✅ 每次写入后自动备份（尽可能防止“打开空库”）
    if not skip_backup:
        try:
            df_all = load_all_rows()
            _save_backup_csv(df_all)
        except Exception:
            pass

def clear_all_data():
    init_db()
    with db_conn() as conn:
        conn.execute(f"DELETE FROM {TABLE_NAME}")
        conn.commit()
    # 不删除备份：避免误点“清空”后无处恢复

# =========================
# Utils
# =========================
NUM_COLS = ["view", "like", "coin", "favorite", "reply", "danmaku", "share", "fans_delta"]
EXTRA_COLS = ["baseline_for", "data_type"]

def parse_bvid(url_or_bv: str) -> str | None:
    s = (url_or_bv or "").strip()
    m = re.search(r"(BV[0-9A-Za-z]{10})", s)
    return m.group(1) if m else None

def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return int(float(x))
    except Exception:
        return default

def _safe_str(x, default=""):
    try:
        if pd.isna(x):
            return default
        return str(x)
    except Exception:
        return default

def _safe_date(x):
    try:
        if pd.isna(x):
            return pd.NaT
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT

def _norm_mid(x) -> str:
    """mid 统一为纯数字字符串；超长mid视为异常。"""
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"[^\d]", "", s)
    if not s:
        return ""
    if len(s) > 12:
        return ""
    return s

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    zh_alias = {
        "项目": "project",
        "项目名": "project",
        "视频链接": "url",
        "链接": "url",
        "标题": "title",
        "昵称": "owner_name",
        "账号昵称": "owner_name",
        "UP昵称": "owner_name",
        "UP主": "owner_name",
        "UP主名称": "owner_name",
        "发布时间": "pubdate",
        "播放": "view",
        "播放量": "view",
        "点赞": "like",
        "投币": "coin",
        "收藏": "favorite",
        "评论": "reply",
        "弹幕": "danmaku",
        "分享": "share",
        "粉丝增长": "fans_delta",
        "粉丝增量": "fans_delta",
        "BV": "bvid",
        "BV号": "bvid",
        "视频BV": "bvid",
        "视频BV号": "bvid",
        "视频BV链接": "url",
        "BV链接": "url",
        "bvid": "bvid",
        "owner_mid": "owner_mid",
        "mid": "owner_mid",
        "基准归属": "baseline_for",
        "数据类型": "data_type",
        "抓取时间": "fetched_at",
    }

    rename = {}
    for c in df.columns:
        key = str(c).strip()
        if key in zh_alias:
            rename[c] = zh_alias[key]
        else:
            low = key.lower()
            if low in [
                "project","url","bvid","title","owner_name","owner_mid","pubdate",
                "view","like","coin","favorite","reply","danmaku","share","fans_delta",
                "baseline_for","data_type","fetched_at"
            ]:
                rename[c] = low
    df = df.rename(columns=rename)

    if "bvid" not in df.columns and "url" in df.columns:
        df["bvid"] = df["url"].apply(parse_bvid)

    if "bvid" in df.columns:
        df["bvid"] = df["bvid"].apply(lambda x: parse_bvid(x) if isinstance(x, str) else x)
        df["bvid"] = df["bvid"].apply(lambda x: _safe_str(x))

    for col in ["project", "title", "owner_name"]:
        if col not in df.columns:
            df[col] = ""
    for col in EXTRA_COLS:
        if col not in df.columns:
            df[col] = ""

    if "owner_mid" not in df.columns:
        df["owner_mid"] = ""
    df["owner_mid"] = df["owner_mid"].apply(_norm_mid)

    if "pubdate" not in df.columns:
        df["pubdate"] = pd.NaT
    df["pubdate"] = df["pubdate"].apply(_safe_date)

    for col in NUM_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].apply(_safe_int)

    if "fetched_at" not in df.columns:
        df["fetched_at"] = pd.Timestamp.now()
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce").fillna(pd.Timestamp.now())

    keep = set([
        "project","bvid","url","title","pubdate","owner_mid","owner_name",
        "view","like","coin","favorite","reply","danmaku","share","fans_delta",
        "baseline_for","data_type","fetched_at"
    ])
    df = df[[c for c in df.columns if c in keep]].copy()
    if "bvid" in df.columns:
        df = df[df["bvid"].astype(str).str.startswith("BV")]
    return df

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["engagement"] = df["like"] + df["coin"] + df["favorite"] + df["reply"]
    df["engagement_rate"] = np.where(df["view"] > 0, df["engagement"] / df["view"], 0.0)
    df["deep_signal_ratio"] = np.where(
        df["engagement"] > 0, (df["coin"] + df["favorite"]) / df["engagement"], 0.0
    )
    return df

def _record_video_snapshots(df_rows: pd.DataFrame):
    """把当前指标按天写入快照表，用于首日/3日/7日/累计表现对比。"""
    if df_rows is None or df_rows.empty:
        return
    init_db()
    d = normalize_df(df_rows.copy())
    if d.empty:
        return
    d = compute_metrics(d)
    d["fetched_at"] = pd.to_datetime(d.get("fetched_at", pd.Timestamp.now()), errors="coerce").fillna(pd.Timestamp.now())
    d["snapshot_date"] = d["fetched_at"].dt.strftime("%Y-%m-%d")
    d["pubdate"] = pd.to_datetime(d["pubdate"], errors="coerce")

    cols = [
        "project", "bvid", "snapshot_date", "fetched_at", "pubdate", "owner_mid", "owner_name", "title",
        "view", "like", "coin", "favorite", "reply", "danmaku", "share",
        "engagement", "engagement_rate", "deep_signal_ratio", "data_type"
    ]
    for c in cols:
        if c not in d.columns:
            d[c] = None
    d = d[cols].copy()
    d["fetched_at"] = pd.to_datetime(d["fetched_at"], errors="coerce").fillna(pd.Timestamp.now()).dt.strftime("%Y-%m-%d %H:%M:%S")
    d["pubdate"] = pd.to_datetime(d["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    records = []
    for _, r in d.iterrows():
        records.append(tuple(None if pd.isna(v) else v for v in r.tolist()))
    if not records:
        return

    placeholders = ",".join(["?"] * len(cols))
    colnames = ",".join(cols)
    sql = f"INSERT OR REPLACE INTO {SNAPSHOT_TABLE_NAME} ({colnames}) VALUES ({placeholders})"
    with db_conn() as conn:
        conn.executemany(sql, records)
        conn.commit()
    try:
        _save_snapshot_backup_csv(load_snapshots())
    except Exception:
        pass

def load_snapshots() -> pd.DataFrame:
    init_db()
    try:
        with db_conn() as conn:
            df = pd.read_sql_query(f"SELECT * FROM {SNAPSHOT_TABLE_NAME}", conn)
        if df is None or df.empty:
            if os.path.exists(BACKUP_SNAPSHOTS_CSV):
                for enc in ["utf-8-sig", "utf-8", "gbk"]:
                    try:
                        df_restore = pd.read_csv(BACKUP_SNAPSHOTS_CSV, encoding=enc)
                        if df_restore is not None and not df_restore.empty:
                            cols = [
                                "project", "bvid", "snapshot_date", "fetched_at", "pubdate", "owner_mid", "owner_name", "title",
                                "view", "like", "coin", "favorite", "reply", "danmaku", "share",
                                "engagement", "engagement_rate", "deep_signal_ratio", "data_type"
                            ]
                            for c in cols:
                                if c not in df_restore.columns:
                                    df_restore[c] = None
                            records = []
                            for _, r in df_restore[cols].iterrows():
                                records.append(tuple(None if pd.isna(v) else v for v in r.tolist()))
                            if records:
                                placeholders = ",".join(["?"] * len(cols))
                                colnames = ",".join(cols)
                                sql = f"INSERT OR REPLACE INTO {SNAPSHOT_TABLE_NAME} ({colnames}) VALUES ({placeholders})"
                                with db_conn() as conn:
                                    conn.executemany(sql, records)
                                    conn.commit()
                            with db_conn() as conn:
                                df = pd.read_sql_query(f"SELECT * FROM {SNAPSHOT_TABLE_NAME}", conn)
                        break
                    except Exception:
                        continue
            if df is None or df.empty:
                return pd.DataFrame()
        for c in ["view", "like", "coin", "favorite", "reply", "danmaku", "share", "engagement"]:
            if c in df.columns:
                df[c] = df[c].apply(_safe_int)
        for c in ["engagement_rate", "deep_signal_ratio"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df["pubdate"] = pd.to_datetime(df["pubdate"], errors="coerce")
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
        df["owner_mid"] = df["owner_mid"].apply(_norm_mid)
        return df
    except Exception:
        return pd.DataFrame()

def _sort_owner_hist(df_owner: pd.DataFrame) -> pd.DataFrame:
    g = df_owner.copy()
    g["__sort_time"] = g["pubdate"]
    missing = g["__sort_time"].isna()
    g.loc[missing, "__sort_time"] = g.loc[missing, "fetched_at"]
    g = g[pd.notna(g["__sort_time"])].sort_values("__sort_time", ascending=False)
    return g

# =========================
# Performance labels (用于Top/Bottom和表格的“发挥”标签)
# =========================
def perf_label(value: float, baseline_values: np.ndarray, ratio_hi: float, ratio_lo: float, min_n: int) -> str:
    baseline_values = baseline_values[~np.isnan(baseline_values)]
    if len(baseline_values) < min_n:
        return "基准不足"
    med = float(np.median(baseline_values))
    ratio = (value / med) if med > 1e-12 else np.inf
    if ratio >= ratio_hi:
        return "超常发挥"
    if ratio <= ratio_lo:
        return "低于预期"
    return "正常发挥"

def build_owner_cache(df_all: pd.DataFrame) -> dict:
    cache = {}
    for up, g in df_all.groupby("owner_name"):
        cache[up] = _sort_owner_hist(g)
    return cache

def recent_baseline(owner_hist_desc: pd.DataFrame, current_bvid: str, col: str, window_n: int) -> np.ndarray:
    if owner_hist_desc is None or owner_hist_desc.empty:
        return np.array([], dtype=float)
    h = owner_hist_desc[owner_hist_desc["bvid"] != current_bvid]
    if h.empty:
        return np.array([], dtype=float)
    return h.head(window_n)[col].astype(float).to_numpy()

def add_perf_cols(df_show: pd.DataFrame, df_all: pd.DataFrame, window_n: int, min_n: int) -> pd.DataFrame:
    df_show = df_show.copy()
    cache = build_owner_cache(df_all)
    v_labels, er_labels = [], []
    for _, r in df_show.iterrows():
        up = r.get("owner_name", "")
        bvid = r.get("bvid", "")
        owner_hist = cache.get(up, None)
        v_base = recent_baseline(owner_hist, bvid, "view", window_n)
        er_base = recent_baseline(owner_hist, bvid, "engagement_rate", window_n)
        v_labels.append(perf_label(float(r.get("view", 0)), v_base, ratio_hi=1.5, ratio_lo=0.7, min_n=min_n))
        er_labels.append(perf_label(float(r.get("engagement_rate", 0.0)), er_base, ratio_hi=1.3, ratio_lo=0.75, min_n=min_n))
    df_show["播放表现"] = v_labels
    df_show["互动率表现"] = er_labels
    return df_show

# =========================
# ✅ WBI 签名
# =========================
_MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32,
    15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19,
    29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61,
    26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63,
    57, 62, 11, 36, 20, 34, 44, 52,
]

def _get_mixin_key(img_key: str, sub_key: str) -> str:
    s = img_key + sub_key
    return "".join([s[i] for i in _MIXIN_KEY_ENC_TAB])[:32]

@st.cache_data(ttl=60*5)  # ✅ 优化：从30分钟改为5分钟，避免WBI签名过期
def _get_wbi_keys() -> tuple[str, str]:
    nav = "https://api.bilibili.com/x/web-interface/nav"
    r = requests.get(nav, headers=HEADERS, timeout=10)
    j = r.json()
    wbi_img = (j.get("data") or {}).get("wbi_img") or {}
    img_url = wbi_img.get("img_url", "")
    sub_url = wbi_img.get("sub_url", "")
    img_key = img_url.split("/")[-1].split(".")[0]
    sub_key = sub_url.split("/")[-1].split(".")[0]
    if not img_key or not sub_key:
        raise RuntimeError("未获取到 WBI img_key/sub_key")
    return img_key, sub_key

def _get_wbi_keys_by_session(sess: requests.Session | None = None) -> tuple[str, str]:
    """
    优先用缓存的全局 WBI key；失败时用当前 Session 重试。
    这样当用户填了代理/Cookie 时，WBI 签名也能走同一条网络链路。
    """
    try:
        return _get_wbi_keys()
    except Exception:
        if sess is None:
            raise
        nav = "https://api.bilibili.com/x/web-interface/nav"
        r = sess.get(nav, headers={"Referer": "https://www.bilibili.com/"}, timeout=10)
        j = r.json()
        wbi_img = (j.get("data") or {}).get("wbi_img") or {}
        img_url = wbi_img.get("img_url", "")
        sub_url = wbi_img.get("sub_url", "")
        img_key = img_url.split("/")[-1].split(".")[0]
        sub_key = sub_url.split("/")[-1].split(".")[0]
        if not img_key or not sub_key:
            raise RuntimeError("未获取到 WBI img_key/sub_key")
        return img_key, sub_key

def _wbi_sign(params: dict, sess: requests.Session | None = None) -> dict:
    img_key, sub_key = _get_wbi_keys_by_session(sess)
    mixin_key = _get_mixin_key(img_key, sub_key)

    params = {k: v for k, v in params.items() if v is not None}
    params["wts"] = int(time.time())

    def _filter(v):
        return re.sub(r"[!'()*]", "", str(v))

    sorted_items = sorted((k, _filter(v)) for k, v in params.items())
    # WBI 更接近 encodeURIComponent：空格用 %20，而不是 application/x-www-form-urlencoded 的 +
    query = urllib.parse.urlencode(sorted_items, quote_via=urllib.parse.quote)
    w_rid = hashlib.md5((query + mixin_key).encode("utf-8")).hexdigest()
    params["w_rid"] = w_rid
    return params

# =========================
# B站抓取
# =========================
def _sleep_jitter(base: float = 1.0):
    """✅ 优化：增加随机等待时间，降低风控风险"""
    try:
        base = float(base)
    except Exception:
        base = 1.0
    # 增加基础延迟和随机抖动，避免触发风控
    time.sleep(max(0.5, base) + random.uniform(0.3, 1.0))


def _apply_cookie_to_session(sess: requests.Session, cookie: str = "") -> requests.Session:
    cookie = (cookie or "").strip()
    if cookie:
        # 只在用户主动提供时使用；避免在代码里硬编码敏感 cookie
        sess.headers.update({"Cookie": cookie})
    return sess


def _normalize_proxy_url(proxy: str = "") -> str:
    """规范化代理地址；用户填 127.0.0.1:7890 时自动补 http://。"""
    proxy = (proxy or "").strip()
    if not proxy:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", proxy):
        proxy = "http://" + proxy
    return proxy


def _apply_proxy_to_session(sess: requests.Session, proxy: str = "") -> requests.Session:
    proxy = _normalize_proxy_url(proxy)
    if proxy:
        sess.proxies.update({"http": proxy, "https": proxy})
    return sess


def _seed_bili_guest_cookies(sess: requests.Session) -> None:
    """
    ✅ 优化：获取更完整的访客Cookie，包括buvid系列
    B站部分空间/搜索接口对没有 buvid 的纯脚本请求更容易返回空或风控。
    这里主动获取公开访客指纹 Cookie；失败不阻断主流程。
    """
    try:
        # 设置基础cookie
        sess.cookies.set("b_nut", str(int(time.time())), domain=".bilibili.com")
        
        # 获取访客指纹
        r = sess.get("https://api.bilibili.com/x/frontend/finger/spi", timeout=8)
        j = r.json()
        data = j.get("data") or {}
        
        # 设置buvid系列cookie
        buvid3 = data.get("b_3") or data.get("buvid3")
        buvid4 = data.get("b_4") or data.get("buvid4")
        
        if buvid3:
            sess.cookies.set("buvid3", str(buvid3), domain=".bilibili.com")
        if buvid4:
            sess.cookies.set("buvid4", str(buvid4), domain=".bilibili.com")
        
        # ✅ 新增：生成必要的cookie（如果没有的话）
        if not sess.cookies.get("buvid"):
            import uuid
            buvid = str(uuid.uuid4()).replace("-", "").upper()
            sess.cookies.set("buvid", buvid, domain=".bilibili.com")
            
        if not sess.cookies.get("_uuid"):
            import uuid
            _uuid = str(uuid.uuid4())
            sess.cookies.set("_uuid", _uuid, domain=".bilibili.com")
            
    except Exception:
        pass


def _make_bili_session(referer: str = "https://www.bilibili.com/", cookie: str = "", proxy: str = "") -> requests.Session:
    """
    ✅ 优化：建立更真实的Session，完整模拟浏览器访问路径
    为每次抓取建立带常见浏览器头、可选 Cookie、可选代理的 Session。
    注意：浏览器开 VPN 不等于 Streamlit/Python requests 也走 VPN；本地运行需要显式填代理。
    """
    sess = requests.Session()
    h = HEADERS.copy()
    origin = "https://space.bilibili.com" if "space.bilibili.com" in _safe_str(referer) else "https://www.bilibili.com"
    h.update({
        "Referer": referer,
        "Origin": origin,
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "DNT": "1",
        "Sec-CH-UA": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    })
    sess.headers.update(h)
    
    # 应用用户提供的cookie和代理
    _apply_cookie_to_session(sess, cookie)
    _apply_proxy_to_session(sess, proxy)
    
    # ✅ 新增：更完整的预热流程
    try:
        # 1. 先访问首页，建立基础cookie
        sess.get("https://www.bilibili.com/", timeout=10)
        time.sleep(0.3)
        
        # 2. 获取访客指纹
        _seed_bili_guest_cookies(sess)
        
        # 3. 访问目标空间页（如果referer包含space.bilibili.com）
        if "space.bilibili.com" in referer:
            try:
                sess.get(referer, timeout=10)
                time.sleep(0.2)
            except Exception:
                pass
                
    except Exception:
        # 预热失败不应导致页面崩溃，后续诊断日志会展示具体接口失败原因
        pass
        
    return sess


def _bili_num_to_int(x, default=0) -> int:
    """兼容 1.2万、3亿、1,234、-- 等 B站常见数字格式。"""
    try:
        if x is None or pd.isna(x):
            return default
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return int(x)
        s = str(x).strip().replace(",", "")
        if not s or s in ["--", "-", "nan", "None"]:
            return default
        m = re.search(r"([\d.]+)\s*([万亿wWkK]?)", s)
        if not m:
            return default
        num = float(m.group(1))
        unit = m.group(2)
        if unit in ["万", "w", "W"]:
            num *= 10000
        elif unit == "亿":
            num *= 100000000
        elif unit in ["k", "K"]:
            num *= 1000
        return int(num)
    except Exception:
        return default


def _clean_bili_title(x) -> str:
    s = _safe_str(x)
    s = re.sub(r"<[^>]+>", "", s)
    return s.replace("&quot;", '"').replace("&amp;", "&").strip()


def _extract_aid(v: dict) -> int | None:
    """从 B站不同接口字段中提取 aid，用于 APP 接口只返回 av 号时再补 BV。"""
    if not isinstance(v, dict):
        return None
    for key in ["aid", "avid", "param", "id", "id_str"]:
        raw = v.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if s.isdigit():
            aid = _safe_int(s, default=0)
            return aid if aid > 0 else None
    for key in ["uri", "arcurl", "url", "jump_url"]:
        s = _safe_str(v.get(key, ""))
        m = re.search(r"(?:av|aid=)(\d+)", s, flags=re.IGNORECASE)
        if m:
            aid = _safe_int(m.group(1), default=0)
            return aid if aid > 0 else None
    return None


def _as_video_row(v: dict) -> dict | None:
    """兼容 B站 Web / APP / 页面 JSON 里常见的视频字段。"""
    if not isinstance(v, dict):
        return None

    aid = _extract_aid(v)
    bvid = v.get("bvid") or v.get("bvidStr") or v.get("BV")
    if not bvid:
        uri = _safe_str(v.get("uri") or v.get("arcurl") or v.get("url") or v.get("jump_url") or "")
        bvid = parse_bvid(uri)
    if not bvid:
        return None

    created = (
        v.get("created")
        or v.get("pubdate")
        or v.get("ctime")
        or v.get("ptime")
        or v.get("publish_time")
        or v.get("pub_time")
    )

    pubdate = pd.NaT
    if created is not None and str(created).strip() != "":
        try:
            # 秒级时间戳才走 unit=s；其它字符串交给 pandas 自己识别
            created_i = _safe_int(created, default=-1)
            if created_i >= 1000000000:
                pubdate = pd.to_datetime(created_i, unit="s", errors="coerce")
            else:
                pubdate = pd.to_datetime(created, errors="coerce")
        except Exception:
            pubdate = pd.NaT

    stat = v.get("stat") or v.get("stats") or {}
    return {
        "bvid": bvid,
        "aid": aid,
        "title": _clean_bili_title(v.get("title", "")),
        "pubdate": pubdate,
        "view": _bili_num_to_int(v.get("play", stat.get("view", v.get("view", v.get("view_content", 0))))),
        "like": _bili_num_to_int(v.get("like", stat.get("like", 0))),
        "coin": _bili_num_to_int(v.get("coin", stat.get("coin", 0))),
        "favorite": _bili_num_to_int(v.get("favorite", stat.get("favorite", 0))),
        "reply": _bili_num_to_int(v.get("comment", stat.get("reply", v.get("reply", 0)))),
        "danmaku": _bili_num_to_int(v.get("danmaku", stat.get("danmaku", 0))),
        "share": _bili_num_to_int(v.get("share", stat.get("share", 0))),
    }


def _dedupe_rows(rows: list[dict], n: int) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        bvid = r.get("bvid") if isinstance(r, dict) else None
        if not bvid or bvid in seen:
            continue
        seen.add(bvid)
        out.append(r)
        if len(out) >= n:
            break
    return out


def _log_kol_debug(debug: list | None, source: str, ok: bool, msg: str = "", count: int = 0, code=None):
    if debug is None:
        return
    debug.append({
        "source": source,
        "ok": bool(ok),
        "count": int(count or 0),
        "code": "" if code is None else code,
        "message": _safe_str(msg)[:220],
        "time": pd.Timestamp.now().strftime("%H:%M:%S"),
    })


def _extract_video_rows_from_html(html: str, n: int = 30, cookie: str = "", proxy: str = "", sess: requests.Session | None = None) -> list[dict]:
    """从UP空间页源码里兜底提取 BV 号，再用详情接口补数据。"""
    if not html:
        return []

    bvids = []
    # 先尝试解析 __INITIAL_STATE__ 中的 BV；失败则用正则扫全页
    for bv in re.findall(r"BV[0-9A-Za-z]{10}", html):
        if bv not in bvids:
            bvids.append(bv)
        if len(bvids) >= n:
            break

    rows = []
    detail_sess = sess or _make_bili_session(cookie=cookie, proxy=proxy)
    for bvid in bvids[:n]:
        detail = fetch_video_detail_by_bvid(bvid, sess=detail_sess)
        if detail is not None:
            rows.append({
                "bvid": bvid,
                "title": detail.get("title", ""),
                "pubdate": detail.get("pubdate", pd.NaT),
                "view": _safe_int(detail.get("view", 0)),
                "reply": _safe_int(detail.get("reply", 0)),
                "like": _safe_int(detail.get("like", 0)),
                "coin": _safe_int(detail.get("coin", 0)),
                "favorite": _safe_int(detail.get("favorite", 0)),
                "danmaku": _safe_int(detail.get("danmaku", 0)),
                "share": _safe_int(detail.get("share", 0)),
            })
        else:
            rows.append({"bvid": bvid, "title": "", "pubdate": pd.NaT, "view": 0, "reply": 0})
        _sleep_jitter(0.25)
    return rows


def _open_space_with_headless_browser(mid: int, wait_sec: float = 4.0) -> tuple[str, dict]:
    """
    可选兜底：如果本机/部署环境装了 selenium + chrome，就无头打开UP主页。
    没有依赖时静默返回空，不影响原有功能。
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except Exception:
        return "", {}

    driver = None
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1366,900")
        options.add_argument(f"--user-agent={HEADERS['User-Agent']}")
        driver = webdriver.Chrome(options=options)
        url = f"https://space.bilibili.com/{mid}/video"
        driver.get(url)
        time.sleep(max(2.5, float(wait_sec)))
        html = driver.page_source or ""
        cookies = {c.get("name"): c.get("value") for c in driver.get_cookies() if c.get("name")}
        return html, cookies
    except Exception:
        return "", {}
    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass


def fetch_video_detail_by_bvid(bvid: str, sess: requests.Session | None = None, cookie: str = "", proxy: str = "") -> dict | None:
    """✅ 优化：获取视频详情，增加重试次数和错误处理"""
    api = "https://api.bilibili.com/x/web-interface/view"
    close_session = False
    if sess is None:
        sess = _make_bili_session(referer=f"https://www.bilibili.com/video/{bvid}", cookie=cookie, proxy=proxy)
        close_session = True

    try:
        # ✅ 优化：增加重试次数到5次，增加错误处理
        for attempt in range(5):
            try:
                r = sess.get(api, params={"bvid": bvid}, headers={"Referer": f"https://www.bilibili.com/video/{bvid}"}, timeout=12)
                
                # ✅ 新增：检查HTTP状态码
                if r.status_code == 412:
                    # 风控，等待更长时间
                    _sleep_jitter(2.0)
                    continue
                elif r.status_code != 200:
                    _sleep_jitter(1.0)
                    continue
                    
                j = r.json()
                
                # ✅ 新增：更详细的错误码处理
                code = j.get("code")
                if code == -352:
                    # 风控，等待更长时间后重试
                    _sleep_jitter(3.0)
                    continue
                elif code == -400:
                    # 请求错误，跳过
                    break
                elif code == -404:
                    # 视频不存在，跳过
                    break
                elif code != 0:
                    _sleep_jitter(1.5)
                    continue
                    
                d = j["data"]
                stat = d.get("stat", {})
                owner = d.get("owner", {})
                return {
                    "bvid": bvid,
                    "title": d.get("title"),
                    "pubdate": pd.to_datetime(d.get("pubdate", 0), unit="s", errors="coerce"),
                    "owner_mid": owner.get("mid"),
                    "owner_name": owner.get("name"),
                    "view": stat.get("view", 0),
                    "like": stat.get("like", 0),
                    "coin": stat.get("coin", 0),
                    "favorite": stat.get("favorite", 0),
                    "reply": stat.get("reply", 0),
                    "danmaku": stat.get("danmaku", 0),
                    "share": stat.get("share", 0),
                }
            except requests.exceptions.Timeout:
                # 超时，等待后重试
                _sleep_jitter(2.0)
                continue
            except requests.exceptions.RequestException:
                # 网络错误
                _sleep_jitter(1.5)
                continue
            except Exception:
                _sleep_jitter(1.0)
                continue
    finally:
        if close_session:
            try:
                sess.close()
            except Exception:
                pass
    return None


def fetch_video_detail_by_aid(aid, sess: requests.Session | None = None, cookie: str = "", proxy: str = "") -> dict | None:
    """APP 投稿接口有时只返回 aid/av 号，这里用详情接口补回 BV 和完整指标。"""
    aid = _safe_int(aid, default=0)
    if aid <= 0:
        return None
    api = "https://api.bilibili.com/x/web-interface/view"
    close_session = False
    if sess is None:
        sess = _make_bili_session(referer=f"https://www.bilibili.com/video/av{aid}", cookie=cookie, proxy=proxy)
        close_session = True

    try:
        for _ in range(3):
            try:
                r = sess.get(api, params={"aid": aid}, headers={"Referer": f"https://www.bilibili.com/video/av{aid}"}, timeout=10)
                j = r.json()
                if j.get("code") != 0:
                    _sleep_jitter(0.6)
                    continue
                d = j["data"]
                stat = d.get("stat", {})
                owner = d.get("owner", {})
                bvid = d.get("bvid") or ""
                if not bvid:
                    return None
                return {
                    "bvid": bvid,
                    "aid": aid,
                    "title": d.get("title"),
                    "pubdate": pd.to_datetime(d.get("pubdate", 0), unit="s", errors="coerce"),
                    "owner_mid": owner.get("mid"),
                    "owner_name": owner.get("name"),
                    "view": stat.get("view", 0),
                    "like": stat.get("like", 0),
                    "coin": stat.get("coin", 0),
                    "favorite": stat.get("favorite", 0),
                    "reply": stat.get("reply", 0),
                    "danmaku": stat.get("danmaku", 0),
                    "share": stat.get("share", 0),
                }
            except Exception:
                _sleep_jitter(0.6)
    finally:
        if close_session:
            try:
                sess.close()
            except Exception:
                pass
    return None


def _extract_bvids_deep(obj, limit: int = 80) -> list[str]:
    """从动态接口的嵌套 JSON 里提取 BV 号。"""
    found = []
    seen = set()

    def walk(x, depth=0):
        if len(found) >= limit or depth > 10:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ["bvid", "bvidStr", "BV"] and isinstance(v, str):
                    bv = parse_bvid(v)
                    if bv and bv not in seen:
                        seen.add(bv)
                        found.append(bv)
                        if len(found) >= limit:
                            return
                walk(v, depth + 1)
        elif isinstance(x, list):
            for item in x:
                walk(item, depth + 1)
                if len(found) >= limit:
                    return
        elif isinstance(x, str):
            for bv in re.findall(r"BV[0-9A-Za-z]{10}", x):
                if bv not in seen:
                    seen.add(bv)
                    found.append(bv)
                    if len(found) >= limit:
                        return

    walk(obj)
    return found


def _detail_to_video_row(detail: dict) -> dict:
    return {
        "bvid": detail.get("bvid", ""),
        "title": detail.get("title", ""),
        "pubdate": detail.get("pubdate", pd.NaT),
        "view": _safe_int(detail.get("view", 0)),
        "like": _safe_int(detail.get("like", 0)),
        "coin": _safe_int(detail.get("coin", 0)),
        "favorite": _safe_int(detail.get("favorite", 0)),
        "reply": _safe_int(detail.get("reply", 0)),
        "danmaku": _safe_int(detail.get("danmaku", 0)),
        "share": _safe_int(detail.get("share", 0)),
    }


def _fetch_vlist_by_dynamic_space(mid: int, n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """
    动态页兜底：部分 UP 的投稿列表接口为空，但动态流仍能看到公开视频。
    抽到 BV 后会再调详情接口校验 owner_mid，避免串号。
    """
    api = "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space"
    rows = []
    seen_bvid = set()
    offset = ""

    for page in range(1, 7):
        if len(rows) >= n:
            break
        params = {
            "host_mid": mid,
            "timezone_offset": -480,
            "platform": "web",
            "features": "itemOpusStyle,listOnlyfans,opusBigCover,onlyfansVote,endFooterHidden",
            "web_location": "333.999",
        }
        if offset:
            params["offset"] = offset
        try:
            r = sess.get(api, params=params, headers={"Referer": f"https://space.bilibili.com/{mid}/dynamic"}, timeout=12)
            j = r.json()
            code = j.get("code", -1)
            data = j.get("data") or {}
            items = data.get("items") or []
            if code != 0 or not items:
                msg = j.get("message") or json.dumps(data, ensure_ascii=False)[:120]
                _log_kol_debug(debug, "dynamic_space", False, f"page={page}: {msg}", 0, code)
                break

            page_bvids = _extract_bvids_deep(items, limit=max(n * 3, 30))
            matched = 0
            for bvid in page_bvids:
                if bvid in seen_bvid:
                    continue
                seen_bvid.add(bvid)
                detail = fetch_video_detail_by_bvid(bvid, sess=sess)
                if detail is not None and _norm_mid(detail.get("owner_mid", "")) == _norm_mid(mid):
                    rows.append(_detail_to_video_row(detail))
                    matched += 1
                    if len(rows) >= n:
                        break
                _sleep_jitter(0.18)

            _log_kol_debug(debug, "dynamic_space", matched > 0, f"page={page}, bvid={len(page_bvids)}, matched_mid={matched}", matched, code)
            if not data.get("has_more"):
                break
            offset = _safe_str(data.get("offset", ""))
            if not offset:
                break
        except Exception as e:
            _log_kol_debug(debug, "dynamic_space", False, f"page={page}: {type(e).__name__}: {e}", 0, "EXC")
            break
        _sleep_jitter(sleep_sec)

    return _dedupe_rows(rows, n)


def _fetch_vlist_by_related_bvids(mid: int, seed_bvids: list[str], n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """
    从已知合作视频出发，读取“相关视频/作者更多”接口。
    只保留详情接口反查 owner_mid 一致的视频，避免推荐视频串号。
    """
    rows = []
    seen = set()
    seed_bvids = [str(x) for x in (seed_bvids or []) if str(x).startswith("BV")]
    if not seed_bvids:
        _log_kol_debug(debug, "related_by_seed", False, "skip: no seed bvid", 0, "NO_SEED")
        return []

    api = "https://api.bilibili.com/x/web-interface/archive/related"
    for seed in seed_bvids[:3]:
        if len(rows) >= n:
            break
        try:
            r = sess.get(api, params={"bvid": seed}, headers={"Referer": f"https://www.bilibili.com/video/{seed}"}, timeout=12)
            j = r.json()
            code = j.get("code", -1)
            data = j.get("data") or []
            if code != 0 or not data:
                msg = j.get("message") or "empty"
                _log_kol_debug(debug, "related_by_seed", False, f"{seed}: {msg}", 0, code)
                continue

            matched = 0
            checked = 0
            for item in data:
                bvid = item.get("bvid") or parse_bvid(_safe_str(item.get("uri") or item.get("arcurl") or item.get("url") or ""))
                if not bvid or bvid in seen:
                    continue
                seen.add(bvid)
                checked += 1
                detail = fetch_video_detail_by_bvid(bvid, sess=sess)
                if detail is not None and _norm_mid(detail.get("owner_mid", "")) == _norm_mid(mid):
                    rows.append(_detail_to_video_row(detail))
                    matched += 1
                    if len(rows) >= n:
                        break
                _sleep_jitter(0.18)
            _log_kol_debug(debug, "related_by_seed", matched > 0, f"{seed}: checked={checked}, matched_mid={matched}", matched, code)
        except Exception as e:
            _log_kol_debug(debug, "related_by_seed", False, f"{seed}: {type(e).__name__}: {e}", 0, "EXC")
        _sleep_jitter(sleep_sec)

    return _dedupe_rows(rows, n)


def _fetch_vlist_by_seed_video_html(mid: int, seed_bvids: list[str], n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """
    从合作视频详情页 HTML 中提取 BV，再用详情接口校验 owner_mid。
    这是最后兜底之一，适合投稿/动态接口都不给列表的账号。
    """
    rows = []
    seen = set()
    seed_bvids = [str(x) for x in (seed_bvids or []) if str(x).startswith("BV")]
    if not seed_bvids:
        _log_kol_debug(debug, "seed_video_html", False, "skip: no seed bvid", 0, "NO_SEED")
        return []

    for seed in seed_bvids[:3]:
        if len(rows) >= n:
            break
        try:
            html = sess.get(f"https://www.bilibili.com/video/{seed}", headers={"Referer": f"https://www.bilibili.com/video/{seed}"}, timeout=12).text
            bvids = []
            for bv in re.findall(r"BV[0-9A-Za-z]{10}", html or ""):
                if bv not in bvids:
                    bvids.append(bv)
                if len(bvids) >= max(60, n * 4):
                    break

            matched = 0
            for bvid in bvids:
                if bvid in seen:
                    continue
                seen.add(bvid)
                detail = fetch_video_detail_by_bvid(bvid, sess=sess)
                if detail is not None and _norm_mid(detail.get("owner_mid", "")) == _norm_mid(mid):
                    rows.append(_detail_to_video_row(detail))
                    matched += 1
                    if len(rows) >= n:
                        break
                _sleep_jitter(0.18)
            _log_kol_debug(debug, "seed_video_html", matched > 0, f"{seed}: html_bv={len(bvids)}, matched_mid={matched}", matched, "")
        except Exception as e:
            _log_kol_debug(debug, "seed_video_html", False, f"{seed}: {type(e).__name__}: {e}", 0, "EXC")
        _sleep_jitter(sleep_sec)

    return _dedupe_rows(rows, n)


def _rows_from_verified_bvids(mid: int, bvids: list[str], n: int, sess: requests.Session, debug: list | None, source: str) -> list[dict]:
    rows = []
    seen = set()
    checked = 0
    for bvid in bvids:
        if len(rows) >= n:
            break
        bvid = parse_bvid(_safe_str(bvid)) or _safe_str(bvid)
        if not bvid or bvid in seen:
            continue
        seen.add(bvid)
        checked += 1
        detail = fetch_video_detail_by_bvid(bvid, sess=sess)
        if detail is not None and _norm_mid(detail.get("owner_mid", "")) == _norm_mid(mid):
            rows.append(_detail_to_video_row(detail))
        _sleep_jitter(0.16)
    _log_kol_debug(debug, source, len(rows) > 0, f"checked={checked}, matched_mid={len(rows)}", len(rows), "")
    return _dedupe_rows(rows, n)


def _fetch_vlist_by_collections(mid: int, n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """
    合集/视频列表兜底：部分老 UP 或分区账号会把视频放在合集/列表里，
    投稿列表接口为空时，这一路仍可能拿到 archives。
    """
    rows = []
    bvids = []
    season_ids = []
    series_ids = []
    base_headers = {"Referer": f"https://space.bilibili.com/{mid}/lists"}

    def add_bvids_from_obj(obj):
        for bv in _extract_bvids_deep(obj, limit=max(120, n * 5)):
            if bv not in bvids:
                bvids.append(bv)

    def collect_ids(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "season_id":
                    sid = _safe_int(v, default=0)
                    if sid > 0 and sid not in season_ids:
                        season_ids.append(sid)
                elif k == "series_id":
                    sid = _safe_int(v, default=0)
                    if sid > 0 and sid not in series_ids:
                        series_ids.append(sid)
                collect_ids(v)
        elif isinstance(obj, list):
            for x in obj:
                collect_ids(x)

    # 1) 主页合集/系列概览，常直接带 archives。
    overview_apis = [
        "https://api.bilibili.com/x/polymer/web-space/home/seasons_series",
        "https://api.bilibili.com/x/polymer/web-space/seasons_series_list",
    ]
    for api in overview_apis:
        if len(rows) >= n:
            break
        for page in range(1, 4):
            params = {
                "mid": mid,
                "page_num": page,
                "page_size": 20,
                "web_location": "333.999",
            }
            try:
                try:
                    signed = _wbi_sign(params, sess=sess)
                except Exception:
                    signed = params
                r = sess.get(api, params=signed, headers=base_headers, timeout=12)
                j = r.json()
                code = j.get("code", -1)
                data = j.get("data") or {}
                if code != 0 or not data:
                    msg = j.get("message") or "empty"
                    _log_kol_debug(debug, "collections_overview", False, f"{api.split('/')[-1]} p{page}: {msg}", 0, code)
                    break
                add_bvids_from_obj(data)
                collect_ids(data)
                _log_kol_debug(debug, "collections_overview", bool(bvids or season_ids or series_ids), f"{api.split('/')[-1]} p{page}: bvids={len(bvids)}, seasons={len(season_ids)}, series={len(series_ids)}", len(bvids), code)
                page_info = ((data.get("items_lists") or {}).get("page")) or data.get("page") or {}
                total = _safe_int(page_info.get("total", 0), default=0)
                if total and page >= max(1, int(np.ceil(total / 20))):
                    break
            except Exception as e:
                _log_kol_debug(debug, "collections_overview", False, f"{api.split('/')[-1]} p{page}: {type(e).__name__}: {e}", 0, "EXC")
                break
            _sleep_jitter(sleep_sec)

    if bvids:
        rows = _dedupe_rows(rows + _rows_from_verified_bvids(mid, bvids, n, sess, debug, "collections_overview_verify"), n)
        if len(rows) >= n:
            return rows[:n]

    # 2) 按 season_id / series_id 展开完整列表。
    for season_id in season_ids[:8]:
        if len(rows) >= n:
            break
        for page in range(1, 4):
            params = {
                "mid": mid,
                "season_id": season_id,
                "sort_reverse": "false",
                "page_num": page,
                "page_size": 30,
                "web_location": "333.999",
            }
            try:
                try:
                    signed = _wbi_sign(params, sess=sess)
                except Exception:
                    signed = params
                r = sess.get("https://api.bilibili.com/x/polymer/web-space/seasons_archives_list", params=signed, headers=base_headers, timeout=12)
                j = r.json()
                code = j.get("code", -1)
                data = j.get("data") or {}
                archives = data.get("archives") or []
                page_bvids = _extract_bvids_deep(archives, limit=80)
                if code == 0 and page_bvids:
                    rows = _dedupe_rows(rows + _rows_from_verified_bvids(mid, page_bvids, n, sess, debug, "season_archives_verify"), n)
                    _log_kol_debug(debug, "season_archives", True, f"season={season_id} p{page}, bvids={len(page_bvids)}", len(page_bvids), code)
                else:
                    msg = j.get("message") or "empty"
                    _log_kol_debug(debug, "season_archives", False, f"season={season_id} p{page}: {msg}", 0, code)
                    break
                total = _safe_int((data.get("page") or {}).get("total", 0), default=0)
                if total and page >= max(1, int(np.ceil(total / 30))):
                    break
            except Exception as e:
                _log_kol_debug(debug, "season_archives", False, f"season={season_id} p{page}: {type(e).__name__}: {e}", 0, "EXC")
                break
            _sleep_jitter(sleep_sec)

    for series_id in series_ids[:8]:
        if len(rows) >= n:
            break
        for page in range(1, 4):
            params = {
                "mid": mid,
                "series_id": series_id,
                "only_normal": "true",
                "sort": "desc",
                "pn": page,
                "ps": 30,
            }
            try:
                r = sess.get("https://api.bilibili.com/x/series/archives", params=params, headers=base_headers, timeout=12)
                j = r.json()
                code = j.get("code", -1)
                data = j.get("data") or {}
                archives = data.get("archives") or []
                page_bvids = _extract_bvids_deep(archives, limit=80)
                if code == 0 and page_bvids:
                    rows = _dedupe_rows(rows + _rows_from_verified_bvids(mid, page_bvids, n, sess, debug, "series_archives_verify"), n)
                    _log_kol_debug(debug, "series_archives", True, f"series={series_id} p{page}, bvids={len(page_bvids)}", len(page_bvids), code)
                else:
                    msg = j.get("message") or "empty"
                    _log_kol_debug(debug, "series_archives", False, f"series={series_id} p{page}: {msg}", 0, code)
                    break
                total = _safe_int((data.get("page") or {}).get("total", 0), default=0)
                if total and page >= max(1, int(np.ceil(total / 30))):
                    break
            except Exception as e:
                _log_kol_debug(debug, "series_archives", False, f"series={series_id} p{page}: {type(e).__name__}: {e}", 0, "EXC")
                break
            _sleep_jitter(sleep_sec)

    return _dedupe_rows(rows, n)


def _fetch_vlist_by_mid_web_wbi(mid: int, n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """Web 端 UP 投稿列表：主路径。"""
    api = "https://api.bilibili.com/x/space/wbi/arc/search"
    space_url = f"https://space.bilibili.com/{mid}/video"
    rows = []
    ps = min(50, max(10, int(n)))

    for pn in range(1, 8):
        if len(rows) >= n:
            break
        params = {
            "mid": mid,
            "pn": pn,
            "ps": ps,
            "tid": 0,
            "keyword": "",
            "order": "pubdate",
            "order_avoided": "true",
            "platform": "web",
            "gaia_source": "main_web",
            "web_location": "1550101",
            "dm_img_list": "[]",
            "dm_img_str": "",
            "dm_cover_img_str": "",
            "dm_img_inter": '{"ds":[],"wh":[0,0,0],"of":[0,0,0]}',
        }
        try:
            signed = _wbi_sign(params, sess=sess)
            r = sess.get(api, params=signed, headers={"Referer": space_url}, timeout=12)
            j = r.json()
            code = j.get("code", -1)
            data = j.get("data") or {}
            vlist = ((data.get("list") or {}).get("vlist")) or []
            if code == 0 and vlist:
                for v in vlist:
                    row = _as_video_row(v)
                    if row:
                        rows.append(row)
                _log_kol_debug(debug, "web_wbi_arc_search", True, f"pn={pn}", len(vlist), code)
            else:
                msg = j.get("message") or data.get("message") or json.dumps(data, ensure_ascii=False)[:120]
                _log_kol_debug(debug, "web_wbi_arc_search", False, f"pn={pn}: {msg}", 0, code)
                if pn == 1:
                    break
        except Exception as e:
            _log_kol_debug(debug, "web_wbi_arc_search", False, f"pn={pn}: {type(e).__name__}: {e}", 0, "EXC")
            if pn == 1:
                break
        _sleep_jitter(sleep_sec)

    return _dedupe_rows(rows, n)


def _fetch_vlist_by_mid_web_old(mid: int, n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """老 Web 投稿接口：保留兜底。"""
    api = "https://api.bilibili.com/x/space/arc/search"
    space_url = f"https://space.bilibili.com/{mid}/video"
    rows = []
    ps = min(50, max(10, int(n)))

    for pn in range(1, 6):
        if len(rows) >= n:
            break
        params = {
            "mid": mid,
            "pn": pn,
            "ps": ps,
            "tid": 0,
            "keyword": "",
            "order": "pubdate",
            "jsonp": "jsonp",
        }
        try:
            r = sess.get(api, params=params, headers={"Referer": space_url}, timeout=12)
            j = r.json()
            code = j.get("code", -1)
            data = j.get("data") or {}
            vlist = ((data.get("list") or {}).get("vlist")) or []
            if code == 0 and vlist:
                for v in vlist:
                    row = _as_video_row(v)
                    if row:
                        rows.append(row)
                _log_kol_debug(debug, "web_old_arc_search", True, f"pn={pn}", len(vlist), code)
            else:
                msg = j.get("message") or json.dumps(data, ensure_ascii=False)[:120]
                _log_kol_debug(debug, "web_old_arc_search", False, f"pn={pn}: {msg}", 0, code)
                if pn == 1:
                    break
        except Exception as e:
            _log_kol_debug(debug, "web_old_arc_search", False, f"pn={pn}: {type(e).__name__}: {e}", 0, "EXC")
            if pn == 1:
                break
        _sleep_jitter(sleep_sec)

    return _dedupe_rows(rows, n)


def _fetch_vlist_by_mid_app_cursor(mid: int, n: int, sess: requests.Session, sleep_sec: float, debug: list | None = None) -> list[dict]:
    """
    APP 端 cursor 投稿接口兜底：无需 WBI。
    这个接口经常只给 aid/param，不给 bvid；旧版会把这些有效视频直接丢掉。
    """
    api_candidates = [
        "https://app.biliapi.com/x/v2/space/archive/cursor",
        "https://app.bilibili.com/x/v2/space/archive/cursor",
    ]
    rows = []
    ps = min(50, max(10, int(n)))
    last_aid = None

    for api in api_candidates:
        rows = []
        last_aid = None
        for page_i in range(1, 8):
            if len(rows) >= n:
                break
            params = {
                "vmid": mid,
                "ps": ps,
                "order": "pubdate",
                "platform": "android",
                "mobi_app": "android",
                "build": 6830300,
                "fnver": 0,
                "fnval": 4048,
                "fourk": 1,
                "ts": int(time.time()),
            }
            if last_aid:
                params["aid"] = last_aid

            try:
                app_headers = {
                    "Referer": f"https://space.bilibili.com/{mid}/video",
                    "User-Agent": "Mozilla/5.0 BiliDroid/7.70.0 (android; Android 12)",
                    "Accept": "application/json, text/plain, */*",
                }
                r = sess.get(api, params=params, headers=app_headers, timeout=12)
                j = r.json()
                code = j.get("code", -1)
                data = j.get("data") or {}
                item = data.get("item") or data.get("items") or []
                if code == 0 and item:
                    for v in item:
                        row = _as_video_row(v)
                        if row:
                            rows.append(row)
                            continue
                        aid = _extract_aid(v)
                        if aid:
                            detail = fetch_video_detail_by_aid(aid, sess=sess)
                            if detail is not None:
                                rows.append({
                                    "bvid": detail.get("bvid", ""),
                                    "aid": aid,
                                    "title": detail.get("title", _clean_bili_title(v.get("title", ""))),
                                    "pubdate": detail.get("pubdate", pd.NaT),
                                    "view": _safe_int(detail.get("view", 0)),
                                    "like": _safe_int(detail.get("like", 0)),
                                    "coin": _safe_int(detail.get("coin", 0)),
                                    "favorite": _safe_int(detail.get("favorite", 0)),
                                    "reply": _safe_int(detail.get("reply", 0)),
                                    "danmaku": _safe_int(detail.get("danmaku", 0)),
                                    "share": _safe_int(detail.get("share", 0)),
                                })
                    last = item[-1] if item else {}
                    last_aid = last.get("param") or last.get("aid") or last.get("id")
                    _log_kol_debug(debug, "app_archive_cursor", True, f"{api.split('/')[2]} page={page_i}", len(item), code)
                    if not data.get("has_next", False) or not last_aid:
                        break
                else:
                    msg = j.get("message") or json.dumps(data, ensure_ascii=False)[:120]
                    _log_kol_debug(debug, "app_archive_cursor", False, f"{api.split('/')[2]} page={page_i}: {msg}", 0, code)
                    break
            except Exception as e:
                _log_kol_debug(debug, "app_archive_cursor", False, f"{api.split('/')[2]} page={page_i}: {type(e).__name__}: {e}", 0, "EXC")
                break

            _sleep_jitter(sleep_sec)

        rows = _dedupe_rows(rows, n)
        if rows:
            return rows

    return []



def _fetch_vlist_by_owner_search(
    mid: int,
    owner_name: str,
    n: int,
    sess: requests.Session,
    sleep_sec: float,
    cookie: str = "",
    proxy: str = "",
    debug: list | None = None,
) -> list[dict]:
    """用 UP 昵称做全站视频搜索，再用详情接口反查 owner_mid，严格过滤串号。"""
    owner_name = _safe_str(owner_name).strip()
    if not owner_name:
        _log_kol_debug(debug, "owner_video_search", False, "skip: empty owner_name", 0, "NO_NAME")
        return []

    api = "https://api.bilibili.com/x/web-interface/search/type"
    rows = []
    seen = set()
    for page in range(1, 5):
        if len(rows) >= n:
            break
        params = {
            "search_type": "video",
            "keyword": owner_name,
            "order": "pubdate",
            "page": page,
            "page_size": 20,
            "platform": "web",
            "web_location": 1430654,
        }
        try:
            try:
                signed = _wbi_sign(params, sess=sess)
            except Exception:
                signed = params
            r = sess.get(api, params=signed, headers={"Referer": "https://search.bilibili.com/"}, timeout=12)
            j = r.json()
            code = j.get("code", -1)
            result = ((j.get("data") or {}).get("result")) or []
            if code != 0 or not result:
                msg = j.get("message") or json.dumps(j.get("data") or {}, ensure_ascii=False)[:120]
                _log_kol_debug(debug, "owner_video_search", False, f"page={page}: {msg}", 0, code)
                if page == 1:
                    break
                continue

            checked = 0
            matched = 0
            for item in result:
                bvid = item.get("bvid") or parse_bvid(_safe_str(item.get("arcurl") or item.get("url") or ""))
                if not bvid or bvid in seen:
                    continue
                seen.add(bvid)
                checked += 1

                item_mid = _norm_mid(item.get("mid") or item.get("owner_mid") or item.get("author_mid") or "")
                if item_mid and item_mid == _norm_mid(mid):
                    row = _as_video_row(item) or {}
                    rows.append({
                        "bvid": bvid,
                        "title": row.get("title") or _clean_bili_title(item.get("title", "")),
                        "pubdate": row.get("pubdate", pd.to_datetime(item.get("pubdate", pd.NaT), unit="s", errors="coerce")),
                        "view": _safe_int(row.get("view", _bili_num_to_int(item.get("play", 0)))),
                        "reply": _safe_int(row.get("reply", _bili_num_to_int(item.get("review", item.get("comment", 0))))),
                        "like": _safe_int(row.get("like", 0)),
                        "coin": _safe_int(row.get("coin", 0)),
                        "favorite": _safe_int(row.get("favorite", _bili_num_to_int(item.get("favorites", 0)))),
                        "danmaku": _safe_int(row.get("danmaku", _bili_num_to_int(item.get("video_review", 0)))),
                        "share": _safe_int(row.get("share", 0)),
                    })
                    matched += 1
                    if len(rows) >= n:
                        break
                    _sleep_jitter(0.12)
                    continue

                detail = fetch_video_detail_by_bvid(bvid, sess=sess, cookie=cookie, proxy=proxy)
                if detail is not None and _norm_mid(detail.get("owner_mid", "")) == _norm_mid(mid):
                    rows.append({
                        "bvid": bvid,
                        "title": detail.get("title", item.get("title", "")),
                        "pubdate": detail.get("pubdate", pd.NaT),
                        "view": _safe_int(detail.get("view", _bili_num_to_int(item.get("play", 0)))),
                        "reply": _safe_int(detail.get("reply", 0)),
                        "like": _safe_int(detail.get("like", 0)),
                        "coin": _safe_int(detail.get("coin", 0)),
                        "favorite": _safe_int(detail.get("favorite", 0)),
                        "danmaku": _safe_int(detail.get("danmaku", 0)),
                        "share": _safe_int(detail.get("share", 0)),
                    })
                    matched += 1
                    if len(rows) >= n:
                        break
                _sleep_jitter(0.18)
            _log_kol_debug(debug, "owner_video_search", matched > 0, f"page={page}, checked={checked}, matched_mid={matched}", matched, code)
        except Exception as e:
            _log_kol_debug(debug, "owner_video_search", False, f"page={page}: {type(e).__name__}: {e}", 0, "EXC")
            if page == 1:
                break
        _sleep_jitter(sleep_sec)
    return _dedupe_rows(rows, n)

def fetch_vlist_by_mid(
    mid: int,
    n: int = 30,
    use_browser_fallback: bool = False,
    sleep_sec: float = 0.8,
    cookie: str = "",
    proxy: str = "",
    owner_name: str = "",
    seed_bvids: list[str] | None = None,
    debug: list | None = None,
) -> list[dict]:
    """
    KOL公开视频列表抓取：多路径聚合，不再“抓到3条就提前返回”。
    目标是尽量补满 N 条，同时每一路都写入诊断日志，方便定位到底卡在哪个接口。
    """
    space_url = f"https://space.bilibili.com/{mid}/video"
    sess = _make_bili_session(referer=space_url, cookie=cookie, proxy=proxy)

    try:
        r = sess.get(space_url, timeout=10)
        _log_kol_debug(debug, "space_warmup", r.status_code == 200, f"HTTP {r.status_code}", 0, r.status_code)
    except Exception as e:
        _log_kol_debug(debug, "space_warmup", False, f"{type(e).__name__}: {e}", 0, "EXC")

    out = []

    def _merge(rows, source_name: str):
        nonlocal out
        before = len(out)
        out = _dedupe_rows(out + (rows or []), n)
        added = len(out) - before
        if added > 0:
            _log_kol_debug(debug, source_name + "_merge", True, f"added={added}, total={len(out)}", added, "MERGE")

    # 1. Web WBI 投稿接口
    _merge(_fetch_vlist_by_mid_web_wbi(mid, n, sess, sleep_sec, debug), "web_wbi_arc_search")
    # 2. 老 Web 投稿接口
    if len(out) < n:
        _merge(_fetch_vlist_by_mid_web_old(mid, n, sess, sleep_sec, debug), "web_old_arc_search")
    # 3. APP cursor 投稿接口
    if len(out) < n:
        _merge(_fetch_vlist_by_mid_app_cursor(mid, n, sess, sleep_sec, debug), "app_archive_cursor")
    # 4. 合集/视频列表兜底
    if len(out) < n:
        _merge(_fetch_vlist_by_collections(mid, n, sess, sleep_sec, debug), "collections")
    # 5. 动态页兜底：部分账号投稿列表为空，但动态页能抽到近期视频
    if len(out) < n:
        _merge(_fetch_vlist_by_dynamic_space(mid, n, sess, sleep_sec, debug), "dynamic_space")
    # 6. 从合作视频出发找“相关/作者更多”视频
    if len(out) < n and seed_bvids:
        _merge(_fetch_vlist_by_related_bvids(mid, seed_bvids, n, sess, sleep_sec, debug), "related_by_seed")
    # 7. 从合作视频详情页 HTML 抽 BV
    if len(out) < n and seed_bvids:
        _merge(_fetch_vlist_by_seed_video_html(mid, seed_bvids, n, sess, sleep_sec, debug), "seed_video_html")
    # 8. 昵称搜索兜底：用于空间投稿接口被风控/为空的账号
    if len(out) < n:
        _merge(_fetch_vlist_by_owner_search(mid, owner_name, n, sess, sleep_sec, cookie=cookie, proxy=proxy, debug=debug), "owner_video_search")
    # 9. 空间 HTML BV 提取兜底
    if len(out) < n:
        try:
            html = sess.get(space_url, timeout=12).text
            html_rows = _extract_video_rows_from_html(html, n=n, cookie=cookie, proxy=proxy, sess=sess)
            _log_kol_debug(debug, "space_html_bv_extract", bool(html_rows), f"html_bv={len(html_rows)}", len(html_rows), "")
            # HTML 抽取也要用详情过滤 owner_mid，避免页面推荐视频串号
            filtered = []
            for row in html_rows:
                bv = row.get("bvid")
                detail = fetch_video_detail_by_bvid(bv, sess=sess, cookie=cookie, proxy=proxy) if bv else None
                if detail is not None and _norm_mid(detail.get("owner_mid", "")) == _norm_mid(mid):
                    filtered.append(row)
            _merge(filtered, "space_html_bv_extract")
        except Exception as e:
            _log_kol_debug(debug, "space_html_bv_extract", False, f"{type(e).__name__}: {e}", 0, "EXC")

    # 10. 可选 Selenium，无依赖时静默跳过
    if len(out) < n and use_browser_fallback:
        html, browser_cookies = _open_space_with_headless_browser(mid)
        if browser_cookies:
            for k, v in browser_cookies.items():
                try:
                    sess.cookies.set(k, v, domain=".bilibili.com")
                except Exception:
                    sess.cookies.set(k, v)
            _merge(_fetch_vlist_by_mid_web_wbi(mid, n, sess, sleep_sec, debug), "selenium_web_wbi")
        if html and len(out) < n:
            html_rows = _extract_video_rows_from_html(html, n=n, cookie=cookie, proxy=proxy, sess=sess)
            _log_kol_debug(debug, "selenium_html_bv_extract", bool(html_rows), f"html_bv={len(html_rows)}", len(html_rows), "")
            _merge(html_rows, "selenium_html_bv_extract")

    if not out:
        _log_kol_debug(debug, "final", False, "all sources returned empty; likely hidden archive, search miss, or interface risk-control", 0, "EMPTY")
    else:
        _log_kol_debug(debug, "final", True, f"final_count={len(out)}", len(out), "OK")
    return out[:n]


def _detail_row_for_project(detail: dict, project: str, url: str = "", data_type: str = "collab", baseline_for: str = "") -> dict:
    """把详情接口结果统一转成数据库行。"""
    bvid = detail.get("bvid", "")
    return {
        "project": project,
        "bvid": bvid,
        "url": _safe_str(url) if _safe_str(url) else f"https://www.bilibili.com/video/{bvid}",
        "title": detail.get("title", ""),
        "pubdate": detail.get("pubdate", pd.NaT),
        "owner_mid": _norm_mid(detail.get("owner_mid", "")),
        "owner_name": detail.get("owner_name", ""),
        "view": _safe_int(detail.get("view", 0)),
        "like": _safe_int(detail.get("like", 0)),
        "coin": _safe_int(detail.get("coin", 0)),
        "favorite": _safe_int(detail.get("favorite", 0)),
        "reply": _safe_int(detail.get("reply", 0)),
        "danmaku": _safe_int(detail.get("danmaku", 0)),
        "share": _safe_int(detail.get("share", 0)),
        "fans_delta": 0,
        "baseline_for": baseline_for,
        "data_type": data_type,
        "fetched_at": pd.Timestamp.now(),
    }

# =========================
# KOL 标注
# =========================
def kol_flag(view_lift: float | None, er_lift: float | None, deep_lift: float | None) -> str:
    def _v(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            return float(x)
        except Exception:
            return None
    v, e, d = _v(view_lift), _v(er_lift), _v(deep_lift)
    if (v is not None and v >= 0.30) or (e is not None and e >= 0.20) or (d is not None and d >= 0.10):
        return "⭐ 合作明显更好"
    if (v is not None and v <= -0.20) or (e is not None and e <= -0.15):
        return "⚠️ 合作偏弱"
    return ""

# =========================
# Sidebar - global settings
# =========================
st.title("B站日常运营数据 Dashboard")
st.sidebar.title("📊 B站运营Dashboard")
# ✅ 展示当前DB、备份位置 & 当前记录数（方便判断是不是“环境重置”）
st.sidebar.caption(f"DB: {DB_PATH}")
st.sidebar.caption(f"Backup: {BACKUP_LATEST_CSV}")
st.sidebar.caption(f"Snapshot Backup: {BACKUP_SNAPSHOTS_CSV}")

st.sidebar.markdown("#### 全局“发挥评价”口径（按KOL自身历史，不按时间）")
baseline_window_n = st.sidebar.slider("基准：取该KOL最近N条视频（按发布时间/抓取时间排序）", 10, 60, 20, step=5)
baseline_min_n = st.sidebar.slider("最低样本数（只与库内条数有关）", 1, 20, 6, step=1)

st.sidebar.divider()

# ✅ 新增：抓取配置部分
with st.sidebar.expander("🔧 抓取配置（建议配置）", expanded=False):
    st.markdown("""**如果抓取失败，请配置以下选项：**
    
1. **Cookie**：在浏览器登录B站后，按F12打开开发者工具，
   切换到Network标签，刷新页面，找到任意请求，
   复制Request Headers中的Cookie值粘贴到此处。
   
2. **代理**：如果你在公司网络环境，需要填写代理地址。
   例如：`127.0.0.1:7890` 或 `http://127.0.0.1:7890`
   
3. **延迟**：增加延迟可以降低被封禁的风险。
    """)
    
    user_cookie = st.text_area(
        "B站Cookie（可选）",
        value="",
        height=80,
        help="登录后的Cookie可以提高抓取成功率。不需要全部Cookie，只需要关键字段如SESSDATA、bili_jct等。"
    )
    
    user_proxy = st.text_input(
        "代理地址（可选）",
        value="",
        placeholder="127.0.0.1:7890",
        help="如果抓取一直失败，可能是公司网络限制，需要配置代理。"
    )
    
    custom_delay = st.slider(
        "自定义抓取延迟（秒）",
        min_value=0.8,
        max_value=3.0,
        value=1.2,
        step=0.1,
        help="延迟越大，越不容易被封禁，但速度会变慢。"
    )
    
    st.info("💡 提示：配置后会在下次抓取时生效。")

with st.sidebar.expander("备份/恢复", expanded=False):
    if st.button("准备导出备份CSV"):
        df_export = load_all_rows()
        if df_export.empty:
            st.info("当前数据库为空，暂无可导出的备份。")
        else:
            st.download_button(
                "⬇️ 导出备份CSV",
                data=df_export.to_csv(index=False).encode("utf-8-sig"),
                file_name="bili_dashboard_backup.csv",
                mime="text/csv"
            )
    uploaded_backup = st.file_uploader("导入备份CSV恢复", type=["csv"])
    if uploaded_backup is not None and st.button("📥 恢复备份到数据库"):
        raw = uploaded_backup.getvalue()
        df_imp = None
        for enc in ["utf-8-sig", "utf-8", "gbk"]:
            try:
                df_imp = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except Exception:
                df_imp = None
        if df_imp is None:
            st.error("恢复失败：CSV读取失败（建议UTF-8编码）。")
        else:
            df_imp = normalize_df(df_imp)
            if "fetched_at" not in df_imp.columns:
                df_imp["fetched_at"] = pd.Timestamp.now()
            df_imp["pubdate"] = pd.to_datetime(df_imp["pubdate"], errors="coerce")
            df_imp["fetched_at"] = pd.to_datetime(df_imp["fetched_at"], errors="coerce").fillna(pd.Timestamp.now())
            df_imp["pubdate"] = df_imp["pubdate"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df_imp["fetched_at"] = df_imp["fetched_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_imp)
            st.success("恢复完成。")
            st.rerun()

with st.sidebar.expander("危险操作：清空全部数据", expanded=False):
    if st.button("🗑️ 清空数据库（不可撤销）"):
        clear_all_data()
        st.success("已清空（备份文件未删除，如需可从备份恢复）。")
        st.rerun()

st.sidebar.divider()

# =========================
# Data input
# =========================
mode = st.sidebar.radio("数据来源", ["粘贴链接/BV采集", "上传CSV/Excel导入"], index=0)

if mode == "粘贴链接/BV采集":
    project = st.sidebar.text_input("项目名（用于归档）", value="未命名项目")
    links = st.sidebar.text_area("粘贴视频链接/ BV号（每行一个）")
    add_btn = st.sidebar.button("➕ 采集并入库（会永久保存）")

    if add_btn:
        items = [x for x in links.splitlines() if x.strip()]
        ok, fail = 0, 0
        rows = []
        
        # ✅ 使用用户配置的cookie和延迟
        for it in items:
            bvid = parse_bvid(it)
            if not bvid:
                fail += 1
                continue
            # ✅ 传递用户配置的cookie和代理
            detail = fetch_video_detail_by_bvid(bvid, cookie=user_cookie, proxy=user_proxy)
            if detail is None:
                fail += 1
                continue
            detail["project"] = project
            detail["url"] = it
            detail["data_type"] = "collab"
            detail["baseline_for"] = ""
            detail["fans_delta"] = 0
            detail["fetched_at"] = pd.Timestamp.now()
            rows.append(detail)
            ok += 1
            # ✅ 使用用户配置的延迟
            time.sleep(custom_delay)

        if rows:
            df_new = normalize_df(pd.DataFrame(rows))
            df_new["pubdate"] = pd.to_datetime(df_new["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_new["fetched_at"] = pd.to_datetime(df_new["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_new)

        st.sidebar.success(f"成功采集 {ok} 条，失败 {fail} 条（已保存+自动备份）")
        st.rerun()

else:
    default_project = st.sidebar.text_input("缺少 project 列时：默认项目名", value="未命名项目")
    uploaded = st.sidebar.file_uploader("选择CSV或Excel文件", type=["csv", "xlsx"])
    import_btn = st.sidebar.button("📥 导入CSV/Excel到仪表盘（会永久保存）")

    if import_btn:
        if not uploaded:
            st.sidebar.error("请先选择一个CSV或Excel文件。")
        else:
            raw = uploaded.getvalue()
            df_import_raw = None
            file_name = getattr(uploaded, "name", "").lower()
            if file_name.endswith(".xlsx"):
                try:
                    sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
                    frames = []
                    for sheet_name, sheet_df in sheets.items():
                        if sheet_df is None or sheet_df.empty:
                            continue
                        sheet_df = sheet_df.copy()
                        sheet_df.columns = [str(c).strip() for c in sheet_df.columns]
                        has_project_col = any(str(c).strip().lower() in ["project", "项目", "项目名"] for c in sheet_df.columns)
                        if not has_project_col:
                            sheet_df["project"] = str(sheet_name)
                        frames.append(sheet_df)
                    df_import_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                except Exception as e:
                    st.sidebar.error(f"Excel读取失败：{e}")
                    df_import_raw = None
            else:
                for enc in ["utf-8-sig", "utf-8", "gbk"]:
                    try:
                        df_import_raw = pd.read_csv(io.BytesIO(raw), encoding=enc)
                        break
                    except Exception:
                        df_import_raw = None

            if df_import_raw is None:
                st.sidebar.error("文件读取失败：CSV建议UTF-8编码；Excel建议使用 .xlsx。")
            else:
                raw_import_count = len(df_import_raw)
                df_csv = normalize_df(df_import_raw)
                dropped_import_count = max(0, raw_import_count - len(df_csv))
                if "project" not in df_csv.columns:
                    df_csv["project"] = default_project
                df_csv["project"] = df_csv["project"].apply(lambda x: _safe_str(x).strip())
                df_csv.loc[df_csv["project"] == "", "project"] = default_project
                if "data_type" not in df_csv.columns:
                    df_csv["data_type"] = "collab"
                if "fetched_at" not in df_csv.columns:
                    df_csv["fetched_at"] = pd.Timestamp.now()

                df_csv["pubdate"] = pd.to_datetime(df_csv["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
                df_csv["fetched_at"] = pd.to_datetime(df_csv["fetched_at"], errors="coerce").fillna(pd.Timestamp.now()).dt.strftime("%Y-%m-%d %H:%M:%S")
                upsert_rows(df_csv)

                st.sidebar.success(f"导入成功：{len(df_csv):,} 行（已保存+自动备份）")
                if dropped_import_count > 0:
                    st.sidebar.warning(f"有 {dropped_import_count} 行未导入：通常是 BV 格式缺失/不完整。B站 BV 号一般应为 BV + 10 位字符，例如 BV1xx411c7mD。")
                st.rerun()

# =========================
# Load data
# =========================
with st.spinner("正在读取本地数据..."):
    df_db = load_all_rows()
df_db = normalize_df(df_db) if not df_db.empty else df_db
st.sidebar.caption(f"Rows: {0 if df_db is None else len(df_db)}")
if df_db is not None and not df_db.empty:
    try:
        _record_video_snapshots(df_db)
    except Exception:
        pass
_BOOT_NOTICE.empty()

if df_db.empty:
    st.info("数据库为空：请在左侧采集或导入。（若你认为数据不应为空：说明部署环境可能重置了磁盘；本应用会优先尝试从 backup/backup_latest.csv 自动恢复）")
    st.stop()

df_db = compute_metrics(df_db)

# =========================
# Project filter (hide baseline project)
# =========================
projects = sorted([p for p in df_db["project"].dropna().unique().tolist()
                   if str(p).strip() != "" and p != BASELINE_PROJECT])
sel_projects = st.sidebar.multiselect("选择项目（筛选展示）", projects, default=projects if projects else None)

df_main = df_db[df_db["project"] != BASELINE_PROJECT].copy()
df_f = df_main[df_main["project"].isin(sel_projects)].copy() if sel_projects else df_main.copy()

# =========================
# Add performance labels
# =========================
df_f = add_perf_cols(df_f, df_db, baseline_window_n, baseline_min_n)

# =========================
# KPI cards
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("总播放", f"{int(df_f['view'].sum()):,}")
c2.metric("总互动(赞+币+藏+评)", f"{int(df_f['engagement'].sum()):,}")
c3.metric("平均互动率", f"{df_f['engagement_rate'].mean()*100:.2f}%")
c4.metric("深度信号占比(币+藏/互动)", f"{df_f['deep_signal_ratio'].mean()*100:.1f}%")

# =========================
# Cross project comparison + Quadrant
# =========================
st.subheader("跨项目对比（项目之间谁更强、谁更稳）")
proj_rows = []
for proj, g in df_f.groupby("project"):
    g2 = g.sort_values("view", ascending=False).copy()
    total_view = int(g2["view"].sum())
    total_eng = int(g2["engagement"].sum())
    video_cnt = int(len(g2))
    up_cnt = int(g2["owner_name"].nunique())

    er_med = float(g2["engagement_rate"].median())
    deep_med = float(g2["deep_signal_ratio"].median())

    er_q1 = float(g2["engagement_rate"].quantile(0.25))
    er_q3 = float(g2["engagement_rate"].quantile(0.75))
    er_iqr = er_q3 - er_q1

    top1_view = int(g2.iloc[0]["view"]) if video_cnt > 0 else 0
    top3_view = int(g2.head(3)["view"].sum()) if video_cnt > 0 else 0
    top1_share = (top1_view / total_view) if total_view > 0 else 0.0
    top3_share = (top3_view / total_view) if total_view > 0 else 0.0

    proj_rows.append({
        "project": proj,
        "视频数": video_cnt,
        "UP数": up_cnt,
        "总播放": total_view,
        "总互动": total_eng,
        "互动率中位数": er_med,
        "深度信号中位数": deep_med,
        "互动率波动(IQR)": er_iqr,
        "Top1播放贡献": top1_share,
        "Top3播放贡献": top3_share,
    })

proj_df = pd.DataFrame(proj_rows).sort_values("总播放", ascending=False)

st.dataframe(
    proj_df.assign(**{
        "互动率中位数": (proj_df["互动率中位数"]*100).map(lambda x: f"{x:.2f}%"),
        "深度信号中位数": (proj_df["深度信号中位数"]*100).map(lambda x: f"{x:.1f}%"),
        "互动率波动(IQR)": (proj_df["互动率波动(IQR)"]*100).map(lambda x: f"{x:.2f}pp"),
        "Top1播放贡献": (proj_df["Top1播放贡献"]*100).map(lambda x: f"{x:.1f}%"),
        "Top3播放贡献": (proj_df["Top3播放贡献"]*100).map(lambda x: f"{x:.1f}%"),
    }),
    use_container_width=True,
    height=260
)

st.markdown("**项目四象限（X=互动率中位数，Y=深度信号中位数）**")
if len(proj_df) >= 2:
    x_med = float(proj_df["互动率中位数"].median())
    y_med = float(proj_df["深度信号中位数"].median())

    fig_q = px.scatter(
        proj_df,
        x="互动率中位数",
        y="深度信号中位数",
        size="总播放",
        text="project",
        hover_data=["视频数","UP数","总播放","Top1播放贡献","Top3播放贡献","互动率波动(IQR)"],
    )
    fig_q.add_vline(x=x_med, line_dash="dash")
    fig_q.add_hline(y=y_med, line_dash="dash")
    fig_q.update_traces(textposition="top center")
    fig_q.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig_q, use_container_width=True)

# =========================
# ✅ 跨项目解读（四象限下方：项目对比）
# =========================
st.subheader("跨项目解读（四象限下方：用于对比不同项目）")
if proj_df.empty:
    st.info("暂无项目数据可解读。")
else:
    p = proj_df.copy()
    p["er"] = p["互动率中位数"]
    p["deep"] = p["深度信号中位数"]
    p["iqr"] = p["互动率波动(IQR)"]
    p["top1"] = p["Top1播放贡献"]
    p["top3"] = p["Top3播放贡献"]

    strongest = p.sort_values(["er","deep"], ascending=False).head(1).iloc[0]
    steadiest = p.sort_values(["iqr","er"], ascending=[True, False]).head(1).iloc[0]
    risky = p.sort_values(["top1","iqr"], ascending=False).head(1).iloc[0]

    lines = []
    lines.append("1）整体结构：当前项目在四象限中呈现差异化分布，可采用不同内容打法与KPI重点。")
    lines.append(f"2）更强项目（互动&沉淀更靠前）：{strongest['project']}（互动率中位数 {strongest['er']*100:.2f}%，深度信号中位数 {strongest['deep']*100:.1f}%）。")
    lines.append(f"3）更稳项目（波动更小）：{steadiest['project']}（互动率波动IQR {steadiest['iqr']*100:.2f}pp）。")
    lines.append(f"4）结构风险提示：{risky['project']} Top1播放贡献 {risky['top1']*100:.1f}%（Top3 {risky['top3']*100:.1f}%），建议补齐腰部内容密度降低单点波动。")
    st.write("\n".join(lines))

# =========================
# 项目内视频表
# =========================
st.divider()
st.subheader("项目内视频表现（按播放排序）")
show_cols = [
    "project","bvid","title","owner_name","pubdate",
    "view","播放表现",
    "engagement_rate","互动率表现",
    "like","coin","favorite","reply",
    "deep_signal_ratio"
]
st.dataframe(df_f[show_cols].sort_values("view", ascending=False), use_container_width=True, height=360)

# =========================
# Top/Bottom 深挖
# =========================
st.subheader("Top / Bottom 深挖（含KOL自身基准判断）")
for proj in (sel_projects if sel_projects else projects):
    d = df_f[df_f["project"] == proj].sort_values("view", ascending=False)
    if d.empty:
        continue
    top = d.iloc[0]
    bottom = d.iloc[-1]

    st.markdown(f"### 项目：{proj}")
    left, right = st.columns(2)

    def render_card(col, row, tag):
        col.markdown(f"**{tag}：{row['title']}**")
        col.caption(f"UP：{row['owner_name']} ｜ BV：{row['bvid']} ｜ 发布：{row['pubdate']}")
        col.metric("播放", f"{int(row['view']):,}", row["播放表现"])
        col.metric("互动率", f"{row['engagement_rate']*100:.2f}%", row["互动率表现"])
        col.write(f"- 赞/币/藏/评：{int(row['like'])}/{int(row['coin'])}/{int(row['favorite'])}/{int(row['reply'])}")
        col.write(f"- 深度信号占比：{row['deep_signal_ratio']*100:.1f}%")

    render_card(left, top, "🔥 最高播放")
    render_card(right, bottom, "🧊 最低播放")

# =========================
# 箱线图
# =========================
st.subheader("互动率分布（项目/UP主快速定位异常）")
fig = px.box(df_f, x="project", y="engagement_rate", points="all", hover_data=["title","owner_name","view"])
st.plotly_chart(fig, use_container_width=True)

# =========================
# ✅ 周报结论（逐项目输出：只评判项目内）
# =========================
st.subheader("周报结论（逐项目输出：只评判项目内）")
projects_for_weekly = sel_projects if (sel_projects and len(sel_projects) > 0) else projects
if not projects_for_weekly:
    st.info("暂无项目可输出周报结论。")
else:
    blocks = []
    idx = 1
    for proj in projects_for_weekly:
        wk = df_f[df_f["project"] == proj].copy()
        if wk.empty:
            continue
        wk = wk.sort_values("view", ascending=False)

        total_view = int(wk["view"].sum())
        total_eng = int(wk["engagement"].sum())
        er_med = float(wk["engagement_rate"].median())
        deep_med = float(wk["deep_signal_ratio"].median())
        video_cnt = int(len(wk))
        up_cnt = int(wk["owner_name"].nunique())

        top = wk.iloc[0]
        bottom = wk.iloc[-1]
        top1_share = float(top["view"]) / total_view if total_view > 0 else 0.0
        top3_share = float(wk.head(3)["view"].sum()) / total_view if total_view > 0 else 0.0
        er_iqr = float(wk["engagement_rate"].quantile(0.75) - wk["engagement_rate"].quantile(0.25))

        lines = []
        lines.append(f"项目{idx}｜【{proj}】")
        lines.append(f"- 产出与规模：{video_cnt} 条内容 / {up_cnt} 位UP，累计播放 {total_view:,}，累计互动 {total_eng:,}。")
        lines.append(f"- 互动质量：互动率中位数 {er_med*100:.2f}%（波动IQR {er_iqr*100:.2f}pp），深度信号中位数 {deep_med*100:.1f}%。")
        lines.append(f"- 高表现样本：最高播放《{top['title']}》{int(top['view']):,} 播放，互动率 {top['engagement_rate']*100:.2f}%，具备可复用抓手。")
        lines.append(f"- 待优化样本：最低播放《{bottom['title']}》{int(bottom['view']):,} 播放，建议从封面/标题信息密度与评论区互动引导做轻量优化，抬升底盘。")
        lines.append(f"- 结构观察：Top1贡献 {top1_share*100:.1f}%（Top3 {top3_share*100:.1f}%），后续通过复用高表现模板+补齐腰部内容，降低单点波动。")

        blocks.append("\n".join(lines))
        idx += 1

    st.write("\n\n".join(blocks))

# =========================
# 保留：全局自动解读（原模块保留）
# =========================
st.subheader("全局自动解读（原模块保留）")
best = df_f.sort_values("view", ascending=False).iloc[0]
worst = df_f.sort_values("view", ascending=True).iloc[0]
insights = []
insights.append(
    f"1）本期最高播放来自《{best['title']}》（{int(best['view']):,} 播放，{best['播放表现']}），互动率 {best['engagement_rate']*100:.2f}%（{best['互动率表现']}）。"
)
insights.append(
    f"2）最低播放为《{worst['title']}》（{int(worst['view']):,} 播放，{worst['播放表现']}），互动率 {worst['engagement_rate']*100:.2f}%（{worst['互动率表现']}）。建议检查封面/标题信息密度与投放时段，并在评论区做更强的互动引导。"
)
if df_f["deep_signal_ratio"].mean() < 0.35:
    insights.append("3）整体深度信号偏低（币+藏在互动中的占比不高），说明内容更多是“路过型热度”，建议强化：价值点前置、结尾引导收藏/投币、增加系列化承诺。")
else:
    insights.append("3）整体深度信号健康（币+藏占比高），说明内容具备沉淀属性，可考虑围绕该方向做系列化与固定栏目节奏。")
st.write("\n".join(insights))



# =========================
# KOL scoring + visual layer
# =========================
KOL_GRADE_ORDER = ["A 重点续约", "B 可继续", "C 待观察", "D 谨慎投放", "E 无法判断"]
KOL_GRADE_COLORS = {
    "A 重点续约": "#009E73",  # green
    "B 可继续": "#0072B2",    # blue
    "C 待观察": "#E69F00",    # orange
    "D 谨慎投放": "#D55E00",  # vermillion/red
    "E 无法判断": "#7F7F7F",  # gray
}
KOL_GRADE_DESC = {
    "A 重点续约": "播放/互动/沉淀综合强，可优先续投或放量",
    "B 可继续": "整体可投，但仍需看素材或样本可靠性",
    "C 待观察": "有单项亮点或样本不稳，适合小额验证",
    "D 谨慎投放": "综合偏弱或互动质量不足，建议收缩",
    "E 无法判断": "没有有效基准，不做投放强判断",
}


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _lift_value(collab: float, base: float) -> float:
    base = _safe_float(base)
    collab = _safe_float(collab)
    if np.isnan(base) or base <= 0 or np.isnan(collab):
        return np.nan
    return collab / base - 1.0


def _fmt_lift(x) -> str:
    x = _safe_float(x)
    if np.isnan(x):
        return "-"
    return f"{x * 100:.1f}%"


def _fmt_rate(x) -> str:
    x = _safe_float(x)
    if np.isnan(x):
        return "-"
    return f"{x * 100:.2f}%"


def _score_component(lift: float, pos_cap: float = 1.2, neg_cap: float = -0.8) -> float:
    """0% 提升=50分；正向封顶，负向扣分。"""
    x = _safe_float(lift)
    if np.isnan(x):
        return 50.0
    if x >= 0:
        return 50.0 + 50.0 * min(x, pos_cap) / pos_cap
    return 50.0 * (1.0 - min(abs(x), abs(neg_cap)) / abs(neg_cap))


def _recommendation(view_lift: float, er_lift: float, deep_lift: float, base_type: str, sample_n: int, min_n: int) -> tuple[str, float]:
    """统一的 ABCDE 分级：推荐等级完全由综合评分/可判断性映射。"""
    if base_type == "无基准":
        return "E 无法判断", np.nan

    view_score = _score_component(view_lift, pos_cap=1.4, neg_cap=-0.9)
    er_score = _score_component(er_lift, pos_cap=0.9, neg_cap=-0.7)
    deep_score = _score_component(deep_lift, pos_cap=0.9, neg_cap=-0.7)

    # 互动质量权重高于播放，避免“高播放低互动”被误判为强推荐。
    score = 0.35 * view_score + 0.45 * er_score + 0.20 * deep_score

    # 样本可靠性折扣，但不创造新的分类；可靠性只进表格，不进图例。
    if base_type == "平台替代基准":
        score = 50.0 + (score - 50.0) * 0.72
        score = min(score, 74.0)
    elif sample_n < min_n:
        score = 50.0 + (score - 50.0) * 0.84
        score = min(score, 74.0)

    # 质量红线：互动明显下滑时限分。
    e = _safe_float(er_lift)
    v = _safe_float(view_lift)
    d = _safe_float(deep_lift)
    if not np.isnan(e) and e <= -0.55:
        score = min(score, 42.0)
    elif not np.isnan(e) and e <= -0.30:
        score = min(score, 58.0)
    if not np.isnan(v) and v <= -0.55:
        score = min(score, 44.0)
    if not np.isnan(d) and d <= -0.55:
        score = min(score, 58.0)

    score = float(np.clip(score, 0, 100))
    if score >= 75:
        grade = "A 重点续约"
    elif score >= 60:
        grade = "B 可继续"
    elif score >= 45:
        grade = "C 待观察"
    else:
        grade = "D 谨慎投放"
    return grade, score


def _build_kol_compare_lib(
    df_all_m: pd.DataFrame,
    collab_projects: list[str],
    baseline_window_n: int,
    baseline_min_n: int,
    name_map: dict | None = None,
    use_proxy_baseline: bool = True,
    baseline_exclude_projects: list[str] | None = None,
) -> pd.DataFrame:
    """生成 KOL 对比库。优先使用 KOL 自身历史；自身历史为0时可用平台替代基准，并明确标注。"""
    if df_all_m is None or df_all_m.empty:
        return pd.DataFrame()
    df_all_m = compute_metrics(normalize_df(df_all_m.copy()))
    df_all_m["owner_mid"] = df_all_m["owner_mid"].apply(_norm_mid)
    collab_set = set(collab_projects or [])
    baseline_exclude_set = set(baseline_exclude_projects if baseline_exclude_projects is not None else (collab_projects or []))
    name_map = name_map or {}

    collab_mid_df = df_all_m[df_all_m["project"].isin(collab_set)].copy()
    collab_mid_df = collab_mid_df[collab_mid_df["owner_mid"].astype(str).str.len() > 0]
    if collab_mid_df.empty:
        return pd.DataFrame()

    proxy_pool = df_all_m[(~df_all_m["project"].isin(baseline_exclude_set)) | (df_all_m["project"] == BASELINE_PROJECT)].copy()
    proxy_pool = proxy_pool[proxy_pool["view"].astype(float) > 0]
    proxy_pool = proxy_pool.drop_duplicates(subset=["owner_mid", "bvid"], keep="last")
    proxy_pool = _sort_owner_hist(proxy_pool).head(max(120, baseline_window_n * 20)) if not proxy_pool.empty else proxy_pool
    fallback_proxy_pool = df_all_m[df_all_m["view"].astype(float) > 0].copy()
    fallback_proxy_pool = fallback_proxy_pool.drop_duplicates(subset=["owner_mid", "bvid"], keep="last")
    fallback_proxy_pool = _sort_owner_hist(fallback_proxy_pool).head(max(200, baseline_window_n * 30)) if not fallback_proxy_pool.empty else fallback_proxy_pool

    rows = []
    for mid, g_collab in collab_mid_df.groupby("owner_mid"):
        up_name = (g_collab["owner_name"].value_counts().index[0]
                   if not g_collab["owner_name"].dropna().empty else name_map.get(mid, ""))
        owner_all = df_all_m[df_all_m["owner_mid"] == mid].copy()
        owner_all = _sort_owner_hist(owner_all)
        collab_bvids = set(g_collab["bvid"].astype(str).tolist())

        own_base = owner_all[(~owner_all["project"].isin(baseline_exclude_set)) | (owner_all["project"] == BASELINE_PROJECT)].copy()
        own_base = own_base[~own_base["bvid"].astype(str).isin(collab_bvids)]
        own_base = own_base[own_base["view"].astype(float) > 0]
        own_base = own_base.drop_duplicates(subset=["bvid"], keep="last")
        own_base = _sort_owner_hist(own_base).head(baseline_window_n)

        if len(own_base) >= baseline_min_n:
            base_pool = own_base
            base_type = "KOL历史基准"
            reliability = "高"
        elif len(own_base) > 0:
            base_pool = own_base
            base_type = "KOL历史小样本"
            reliability = "中"
        elif use_proxy_baseline:
            proxy_base = proxy_pool[proxy_pool["owner_mid"] != mid].copy() if proxy_pool is not None and not proxy_pool.empty else pd.DataFrame()
            if proxy_base.empty and fallback_proxy_pool is not None and not fallback_proxy_pool.empty:
                proxy_base = fallback_proxy_pool[fallback_proxy_pool["owner_mid"] != mid].copy()
            if not proxy_base.empty:
                base_pool = proxy_base.head(baseline_window_n)
                base_type = "平台替代基准"
                reliability = "低"
            else:
                base_pool = pd.DataFrame()
                base_type = "无基准"
                reliability = "无"
        else:
            base_pool = pd.DataFrame()
            base_type = "无基准"
            reliability = "无"

        collab_view = _safe_float(g_collab["view"].median())
        collab_er = _safe_float(g_collab["engagement_rate"].median())
        collab_deep = _safe_float(g_collab["deep_signal_ratio"].median())

        if base_pool.empty:
            base_view = base_er = base_deep = np.nan
        else:
            base_view = _safe_float(base_pool["view"].median())
            base_er = _safe_float(base_pool["engagement_rate"].median())
            base_deep = _safe_float(base_pool["deep_signal_ratio"].median())

        view_lift = _lift_value(collab_view, base_view)
        er_lift = _lift_value(collab_er, base_er)
        deep_lift = _lift_value(collab_deep, base_deep)
        grade, score = _recommendation(view_lift, er_lift, deep_lift, base_type, len(base_pool), baseline_min_n)

        mark = kol_flag(view_lift, er_lift, deep_lift)
        if base_type == "平台替代基准":
            mark = "🟡 平台替代基准"
        elif base_type == "KOL历史小样本":
            mark = mark or "🔎 小样本可读"
        elif base_type == "无基准":
            mark = "⚪ 无基准"

        tags = []
        if not np.isnan(view_lift) and view_lift >= 0.30:
            tags.append("热度拉升")
        if not np.isnan(er_lift) and er_lift >= 0.20:
            tags.append("互动增强")
        if not np.isnan(deep_lift) and deep_lift >= 0.10:
            tags.append("沉淀提升")
        if not tags:
            tags.append("常规")

        if np.isnan(view_lift) or np.isnan(er_lift):
            quadrant = "无法判断"
        elif view_lift >= 0 and er_lift >= 0:
            quadrant = "高播放+高互动"
        elif view_lift >= 0 and er_lift < 0:
            quadrant = "高播放+低互动"
        elif view_lift < 0 and er_lift >= 0:
            quadrant = "低播放+高互动"
        else:
            quadrant = "低播放+低互动"

        rows.append({
            "owner_mid": mid,
            "KOL/UP主": up_name,
            "推荐等级": grade,
            "综合评分": score,
            "标注": mark,
            "合作视频数": int(len(g_collab)),
            "基准样本数": int(len(base_pool)),
            "基准类型": base_type,
            "基准可靠性": reliability,
            "象限": quadrant,
            "标签": "、".join(tags),
            "KOL画像一句话": f"{'热度拉升' if '热度拉升' in tags else '热度稳定'} + {'互动增强' if '互动增强' in tags else '互动常规'} + {'沉淀提升' if '沉淀提升' in tags else '沉淀一般'}",
            "合作播放中位数": int(collab_view) if not np.isnan(collab_view) else 0,
            "基准播放中位数": int(base_view) if not np.isnan(base_view) else 0,
            "播放提升值": view_lift,
            "播放提升": _fmt_lift(view_lift),
            "合作互动率值": collab_er,
            "基准互动率值": base_er,
            "合作互动率中位数": _fmt_rate(collab_er),
            "基准互动率中位数": _fmt_rate(base_er),
            "互动率提升值": er_lift,
            "互动率提升": _fmt_lift(er_lift),
            "合作深度信号值": collab_deep,
            "基准深度信号值": base_deep,
            "合作深度信号中位数": _fmt_rate(collab_deep),
            "基准深度信号中位数": _fmt_rate(base_deep),
            "深度信号提升值": deep_lift,
            "深度信号提升": _fmt_lift(deep_lift),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["推荐等级"] = pd.Categorical(out["推荐等级"], categories=KOL_GRADE_ORDER, ordered=True)
    return out.sort_values(["推荐等级", "综合评分"], ascending=[True, False])


def _cap_lift_for_plot(x, lo=-1.0, hi=3.0):
    x = _safe_float(x)
    if np.isnan(x):
        return np.nan
    return float(np.clip(x, lo, hi))


def _render_grade_cards(d: pd.DataFrame):
    counts = d["推荐等级"].astype(str).value_counts().to_dict()
    cols = st.columns(5)
    for col, grade in zip(cols, KOL_GRADE_ORDER):
        col.metric(grade, int(counts.get(grade, 0)))
        col.caption(KOL_GRADE_DESC[grade])


def _render_kol_visuals(lib: pd.DataFrame):
    if lib is None or lib.empty:
        st.info("暂无KOL结果可视化。")
        return
    d = lib.copy()
    d["推荐等级"] = d["推荐等级"].astype(str)
    for c in ["播放提升值", "互动率提升值", "深度信号提升值", "综合评分", "合作播放中位数"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    st.markdown("**统一分级口径（所有图表只使用这一套 ABCDE 颜色）**")
    _render_grade_cards(d)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KOL总数", f"{len(d):,}")
    c2.metric("A/B可继续", f"{int(d['推荐等级'].isin(['A 重点续约', 'B 可继续']).sum()):,}")
    c3.metric("C待观察", f"{int((d['推荐等级'] == 'C 待观察').sum()):,}")
    c4.metric("D/E谨慎或无法判断", f"{int(d['推荐等级'].isin(['D 谨慎投放', 'E 无法判断']).sum()):,}")

    st.markdown("**投放四象限：散点只看分布，名单看下方象限卡片**")
    plot_df = d.dropna(subset=["播放提升值", "互动率提升值"]).copy()
    if not plot_df.empty:
        plot_df["播放提升显示值"] = plot_df["播放提升值"].apply(lambda x: _cap_lift_for_plot(x, -1, 3))
        plot_df["互动率提升显示值"] = plot_df["互动率提升值"].apply(lambda x: _cap_lift_for_plot(x, -1, 2))
        plot_df["气泡大小"] = np.log1p(plot_df["合作播放中位数"].clip(lower=0))
        plot_df["关键标注"] = ""
        key_idx = set(plot_df.sort_values("综合评分", ascending=False).head(4).index.tolist())
        key_idx.update(plot_df[plot_df["推荐等级"].isin(["A 重点续约", "D 谨慎投放"])].sort_values("综合评分", ascending=False).head(4).index.tolist())
        for idx in list(key_idx)[:8]:
            if idx in plot_df.index:
                plot_df.loc[idx, "关键标注"] = plot_df.loc[idx, "KOL/UP主"]

        fig = px.scatter(
            plot_df,
            x="播放提升显示值",
            y="互动率提升显示值",
            size="气泡大小",
            color="推荐等级",
            color_discrete_map=KOL_GRADE_COLORS,
            category_orders={"推荐等级": KOL_GRADE_ORDER},
            text="关键标注",
            hover_name="KOL/UP主",
            hover_data={
                "推荐等级": True,
                "综合评分": ':.1f',
                "基准类型": True,
                "基准可靠性": True,
                "播放提升": True,
                "互动率提升": True,
                "深度信号提升": True,
                "合作播放中位数": True,
                "播放提升显示值": False,
                "互动率提升显示值": False,
                "气泡大小": False,
                "关键标注": False,
            },
            height=620,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_vrect(x0=0, x1=3, y0=0, y1=2, fillcolor="#009E73", opacity=0.06, line_width=0)
        fig.update_traces(textposition="top center", marker=dict(opacity=0.78, line=dict(width=1, color="white")))
        fig.update_layout(
            xaxis_title="播放提升（显示截断：-100% ~ +300%；真实值见悬浮）",
            yaxis_title="互动率提升（显示截断：-100% ~ +200%；真实值见悬浮）",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            legend_title_text="推荐等级",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("颜色只代表推荐等级；基准可靠性仅在悬浮与表格中展示，不参与颜色/图例。")

    st.markdown("**四象限名单（用于汇报理解，不靠散点标签硬挤）**")
    quad_order = [
        ("高播放+高互动", "优先续投/放量"),
        ("高播放+低互动", "有曝光但质量弱，谨慎放量"),
        ("低播放+高互动", "素材或投放匹配可复盘"),
        ("低播放+低互动", "减少投放或重做素材"),
    ]
    cols = st.columns(4)
    for col, (quad, hint) in zip(cols, quad_order):
        sub = d[d["象限"] == quad].sort_values("综合评分", ascending=False).head(10)
        col.markdown(f"**{quad}**")
        col.caption(hint)
        col.metric("数量", int((d["象限"] == quad).sum()))
        if sub.empty:
            col.write("—")
        else:
            lines = [f"{r['KOL/UP主']}｜{r['推荐等级'][0]}｜{_safe_float(r['综合评分'], 0):.1f}" for _, r in sub.iterrows()]
            col.write("\n".join([f"- {x}" for x in lines]))

    st.markdown("**KOL综合评分排行（0-100分；颜色=ABCDE推荐等级）**")
    rank = d.dropna(subset=["综合评分"]).sort_values("综合评分", ascending=False).head(30).copy()
    if rank.empty:
        st.info("暂无可排名KOL。")
    else:
        fig_rank = px.bar(
            rank.sort_values("综合评分", ascending=True),
            x="综合评分",
            y="KOL/UP主",
            orientation="h",
            color="推荐等级",
            color_discrete_map=KOL_GRADE_COLORS,
            category_orders={"推荐等级": KOL_GRADE_ORDER},
            text="综合评分",
            hover_data=["基准类型", "基准可靠性", "播放提升", "互动率提升", "深度信号提升", "象限"],
            height=max(520, min(920, 28 * len(rank) + 160)),
        )
        fig_rank.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_rank.update_layout(xaxis_range=[0, 105], legend_title_text="推荐等级", yaxis_title="KOL/UP主")
        st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("**投放建议分层（统一ABCDE）**")
    cols = st.columns(5)
    for col, grade in zip(cols, KOL_GRADE_ORDER):
        names = d[d["推荐等级"] == grade].sort_values("综合评分", ascending=False)["KOL/UP主"].head(8).tolist()
        col.markdown(f"**{grade}**")
        col.caption(KOL_GRADE_DESC[grade])
        col.write("\n".join([f"- {x}" for x in names]) if names else "—")


def _existing_personal_base_pool(df_all: pd.DataFrame, mid: str, collab_projects: list[str], collab_bvids: set[str]) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame()
    d = df_all.copy()
    d["owner_mid"] = d["owner_mid"].apply(_norm_mid)
    owner_all = d[d["owner_mid"] == _norm_mid(mid)].copy()
    if owner_all.empty:
        return owner_all
    base_pool = owner_all[
        (owner_all["project"] == BASELINE_PROJECT) |
        (~owner_all["project"].isin(set(collab_projects or [])))
    ].copy()
    base_pool = base_pool[~base_pool["bvid"].astype(str).isin(collab_bvids)]
    base_pool = base_pool[base_pool["view"].astype(float) > 0]
    base_pool = base_pool.drop_duplicates(subset=["bvid"], keep="last")
    return _sort_owner_hist(base_pool)


# =========================================================
# KOL module（按 owner_mid，对齐+补齐+标注+导出）
# =========================================================
st.divider()
st.subheader("KOL合作资料库（独立模块：标注合作是否优于平时｜按owner_mid对齐）")

with st.expander("KOL模块设置", expanded=False):
    collab_projects = st.multiselect("哪些项目算合作项目", projects, default=sel_projects if sel_projects else projects)
    _candidate_target_projects = list(collab_projects or [])
    _sidebar_target_projects = [p for p in (sel_projects if sel_projects else []) if p in _candidate_target_projects]
    if _sidebar_target_projects and len(_sidebar_target_projects) < len(projects):
        _default_target_projects = _sidebar_target_projects
    else:
        _default_target_projects = []
        try:
            _pdate = df_db[df_db["project"].isin(_candidate_target_projects)].copy()
            _pdate["pubdate"] = pd.to_datetime(_pdate["pubdate"], errors="coerce")
            _pdate = _pdate.dropna(subset=["pubdate"])
            if not _pdate.empty:
                _latest_project = str(_pdate.sort_values("pubdate").iloc[-1]["project"])
                if _latest_project in _candidate_target_projects:
                    _default_target_projects = [_latest_project]
        except Exception:
            _default_target_projects = []
        if not _default_target_projects and _candidate_target_projects:
            _default_target_projects = [_candidate_target_projects[-1]]
    target_projects = st.multiselect(
        "当前要评估/立项的项目（只排除这些项目；旧合作项目会作为历史参考）",
        _candidate_target_projects,
        default=_default_target_projects,
        help="例如现在看“刺客信条-黑旗重置”，这里只选它。旧项目不要选到这里，否则旧合作视频也会被排除出个人基准。"
    )
    kol_view_scope = st.radio(
        "KOL视觉总览/对比表统计范围",
        ["当前评估项目", "全部合作KOL库"],
        index=0,
        horizontal=True,
        help="当前评估项目用于本次立项/投放决策；全部合作KOL库用于盘点所有历史合作UP主数量。"
    )
    fetch_n = st.slider("补齐基准：每个KOL抓取最近N条公开视频", 10, 80, 30, step=5)
    skip_enough_baseline = st.checkbox("补齐时跳过已达最低样本数的KOL（推荐，避免云端超时）", value=True)
    max_mids_per_run = st.slider("本次最多处理不足KOL数（可重复点击继续补齐）", 5, 80, 20, step=5)
    sleep_sec = st.slider("抓取间隔（防限流）", 0.2, 2.0, 0.8, step=0.1)
    use_browser_fallback = st.checkbox("启用 Selenium 无头浏览器兜底（本机需安装 Chrome/Driver；一般不用开）", value=False)
    show_kol_quality_hint = st.checkbox("显示数据质量提示（缺mid/异常mid/抓取诊断）", value=True)
    use_proxy_baseline = st.checkbox("无个人历史基准时，生成表使用平台替代基准（会明确标注）", value=True)
    
    # ✅ 修改：使用Sidebar中定义的全局配置
    # 注意：user_cookie, user_proxy 在 Sidebar 的“抓取配置”中定义
    try:
        bili_cookie = user_cookie
    except NameError:
        bili_cookie = os.environ.get("BILI_COOKIE", "")
    
    try:
        bili_proxy = user_proxy
    except NameError:
        bili_proxy = os.environ.get("BILI_PROXY", "")

cA, cB, cC, cD = st.columns([1, 1, 1, 2])
with cA:
    btn_fill_all = st.button("🧲 一键补齐所有合作KOL基准（写入__BASELINE__）")
with cB:
    btn_fill_zero = st.button("⚡ 只补齐0样本KOL（新项目推荐）")
with cC:
    btn_build_kol = st.button("📚 生成KOL对比表（含视觉总览）")
with cD:
    st.caption("新项目导入后：优先点“只补齐0样本KOL”；仍为0时再填Cookie重试。")

if collab_projects:
    active_projects = list(target_projects) if target_projects else list(collab_projects)
    if target_projects and set(active_projects) != set(collab_projects):
        st.info(
            "当前KOL评估只针对："
            + "、".join(active_projects)
            + "。其他已选合作项目会作为历史参考参与个人基准。"
        )
    elif len(collab_projects) > 1:
        st.warning("当前没有单独指定评估项目，系统会把所有合作项目都当作当前项目；旧合作视频不会计入个人基准。建议在设置里只选择本次要评估的项目。")

    collab_df = df_db[df_db["project"].isin(active_projects)].copy()
    collab_df["owner_mid"] = collab_df["owner_mid"].apply(_norm_mid)

    valid_mid_df = collab_df[collab_df["owner_mid"].astype(str).str.len() > 0].copy()
    invalid_mid_cnt = int((collab_df["owner_mid"].astype(str).str.len() == 0).sum())
    all_collab_df = df_db[df_db["project"].isin(collab_projects)].copy()
    all_collab_df["owner_mid"] = all_collab_df["owner_mid"].apply(_norm_mid)
    all_valid_mid_df = all_collab_df[all_collab_df["owner_mid"].astype(str).str.len() > 0].copy()
    all_invalid_mid_cnt = int((all_collab_df["owner_mid"].astype(str).str.len() == 0).sum())

    st.caption(
        f"合作UP主数：{valid_mid_df['owner_mid'].nunique()}（可识别mid）"
        f"｜缺/异常mid合作视频：{invalid_mid_cnt}"
        f"｜合作视频数：{len(collab_df)}"
    )
    inv_cols = st.columns(4)
    inv_cols[0].metric("当前评估项目KOL", f"{valid_mid_df['owner_mid'].nunique():,}")
    inv_cols[1].metric("全部合作KOL库", f"{all_valid_mid_df['owner_mid'].nunique():,}")
    inv_cols[2].metric("当前项目视频数", f"{len(collab_df):,}")
    inv_cols[3].metric("全库缺/异常mid视频", f"{all_invalid_mid_cnt:,}")
    if kol_view_scope == "当前评估项目" and all_valid_mid_df["owner_mid"].nunique() > valid_mid_df["owner_mid"].nunique():
        st.caption("提示：当前页的视觉总览默认只看本次评估项目；如需盘点全部历史合作UP主，请在设置里把统计范围切到“全部合作KOL库”。")

    if show_kol_quality_hint:
        bad_rows = collab_df[collab_df["owner_mid"].astype(str).str.len() == 0].copy()
        if not bad_rows.empty:
            st.warning(f"发现 {len(bad_rows)} 条合作视频 owner_mid 缺失/异常。点击“一键补齐”时会先用 BV 详情接口自动修复。")
            st.dataframe(bad_rows[["project","bvid","title","owner_name","owner_mid"]].head(80), use_container_width=True, height=220)
        else:
            st.success("合作视频 owner_mid 看起来正常。")

    name_map = (valid_mid_df.groupby("owner_mid")["owner_name"]
                .agg(lambda s: s.value_counts().index[0]).to_dict()) if not valid_mid_df.empty else {}

    if not valid_mid_df.empty:
        _collab_bvids_preview = set(valid_mid_df["bvid"].astype(str).tolist())
        _zero_mids_preview = []
        for _mid in sorted(valid_mid_df["owner_mid"].unique().tolist()):
            _existing_preview = _existing_personal_base_pool(df_db, _mid, active_projects, _collab_bvids_preview)
            if len(_existing_preview) == 0:
                _zero_mids_preview.append(_mid)
        if _zero_mids_preview:
            st.warning(f"当前有 {len(_zero_mids_preview)} 个KOL没有个人历史基准。新项目导入后这是正常状态，建议先点击“只补齐0样本KOL”。")

    if btn_fill_all or btn_fill_zero:
        progress = st.progress(0)
        status = st.empty()
        debug_rows = []
        rows_to_write = {}
        stat = {
            "collab_repair_total": 0, "collab_repair_ok": 0, "collab_repair_fail": 0,
            "list_fail": 0, "list_empty": 0, "detail_ok": 0, "detail_fail": 0, "vlist_added": 0,
            "collab_refresh_total": 0, "collab_refresh_ok": 0, "collab_refresh_fail": 0,
            "mid_skipped_enough": 0, "mid_need_total": 0, "mid_selected": 0, "baseline_skip_existing": 0
        }

        # ========= A0) 先刷新合作BV详情：修复 CSV 导入缺 owner_mid 的根因 =========
        collab_pairs_all = (df_db[(df_db["project"].isin(collab_projects)) & (df_db["project"] != BASELINE_PROJECT)]
                            [["project","bvid","url","owner_mid"]]
                            .dropna(subset=["project","bvid"])
                            .drop_duplicates()
                            .values
                            .tolist())
        repair_pairs = [x for x in collab_pairs_all if not _norm_mid(x[3])]
        stat["collab_repair_total"] = len(repair_pairs)
        detail_sess = _make_bili_session(cookie=bili_cookie, proxy=bili_proxy)
        total_steps = max(1, len(repair_pairs) + max(1, min(int(max_mids_per_run), valid_mid_df["owner_mid"].nunique())) * 2)
        step = 0

        for proj, bvid, url, _old_mid in repair_pairs:
            step += 1
            progress.progress(min(1.0, step / total_steps))
            status.info(f"正在刷新合作BV详情/修复 owner_mid：{bvid}")
            bvid = str(bvid)
            if not bvid.startswith("BV"):
                stat["collab_repair_fail"] += 1
                continue

            detail = fetch_video_detail_by_bvid(bvid, sess=detail_sess)
            if detail is None:
                stat["collab_repair_fail"] += 1
                _sleep_jitter(float(sleep_sec))
                continue

            row = _detail_row_for_project(detail, project=str(proj), url=url, data_type="collab", baseline_for="")
            rows_to_write[(row["project"], row["bvid"])] = row
            stat["collab_repair_ok"] += 1
            _sleep_jitter(float(sleep_sec))

        if rows_to_write:
            df_repair = normalize_df(pd.DataFrame(list(rows_to_write.values())))
            df_repair["pubdate"] = pd.to_datetime(df_repair["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_repair["fetched_at"] = pd.to_datetime(df_repair["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_repair)
            df_db_work = compute_metrics(normalize_df(load_all_rows()))
        else:
            df_db_work = df_db.copy()

        # 用修复后的 owner_mid 重新计算合作KOL
        collab_df_work = df_db_work[df_db_work["project"].isin(active_projects)].copy()
        collab_df_work["owner_mid"] = collab_df_work["owner_mid"].apply(_norm_mid)
        valid_mid_df_work = collab_df_work[collab_df_work["owner_mid"].astype(str).str.len() > 0].copy()
        name_map_work = (valid_mid_df_work.groupby("owner_mid")["owner_name"]
                         .agg(lambda s: s.value_counts().index[0]).to_dict()) if not valid_mid_df_work.empty else {}

        # ========= A1) 补齐 baseline =========
        existed_baseline = set(df_db_work[df_db_work["project"] == BASELINE_PROJECT]["bvid"].astype(str).tolist())
        baseline_rows_to_write = {}

        collab_bvids_all = set(valid_mid_df_work["bvid"].astype(str).tolist())
        mids_all = sorted(valid_mid_df_work["owner_mid"].unique().tolist())
        mid_need_rows = []
        for mid in mids_all:
            existing_pool = _existing_personal_base_pool(df_db_work, mid, active_projects, collab_bvids_all)
            if bool(btn_fill_zero) and len(existing_pool) > 0:
                stat["mid_skipped_enough"] += 1
                continue
            if bool(skip_enough_baseline) and len(existing_pool) >= int(baseline_min_n):
                stat["mid_skipped_enough"] += 1
                continue
            mid_need_rows.append((int(len(existing_pool)), _safe_str(name_map_work.get(mid, "")), mid))
        mid_need_rows = sorted(mid_need_rows, key=lambda x: (x[0], x[1], x[2]))
        mids = [x[2] for x in mid_need_rows]
        stat["mid_need_total"] = len(mids)
        mids = mids[:int(max_mids_per_run)]
        stat["mid_selected"] = len(mids)

        if not mids:
            if mids_all:
                st.info("所有可识别KOL都已达到最低样本数，本次无需继续抓取。若想补满更多历史，请关闭“跳过已达最低样本数”。")
            else:
                st.error("合作视频详情刷新后仍没有可用 owner_mid：请检查 BV 是否有效，或稍后重试详情接口。")
        else:
            for mid in mids:
                step += 1
                progress.progress(min(1.0, step / total_steps))
                disp = name_map_work.get(mid, "")
                status.info(f"正在抓取KOL基准：{disp or mid}（mid={mid}）")
                per_mid_debug = []
                seed_bvids_this_mid = valid_mid_df_work[valid_mid_df_work["owner_mid"].apply(_norm_mid) == _norm_mid(mid)]["bvid"].astype(str).tolist()
                try:
                    vlist = fetch_vlist_by_mid(
                        int(mid),
                        n=int(fetch_n),
                        use_browser_fallback=bool(use_browser_fallback),
                        sleep_sec=float(sleep_sec),
                        cookie=bili_cookie,
                        proxy=bili_proxy,
                        owner_name=disp,
                        seed_bvids=seed_bvids_this_mid,
                        debug=per_mid_debug,
                    )
                    for d in per_mid_debug:
                        d["owner_mid"] = mid
                        d["KOL/UP主"] = disp
                        debug_rows.append(d)
                except Exception as e:
                    stat["list_fail"] += 1
                    debug_rows.append({"owner_mid": mid, "KOL/UP主": disp, "source": "fetch_vlist_by_mid", "ok": False, "count": 0, "code": "EXC", "message": str(e), "time": pd.Timestamp.now().strftime("%H:%M:%S")})
                    continue

                if not vlist:
                    stat["list_empty"] += 1
                    continue

                existing_bvids_this_mid = set(_existing_personal_base_pool(df_db_work, mid, active_projects, collab_bvids_all)["bvid"].astype(str).tolist())
                for v in vlist:
                    bvid = v.get("bvid", "")
                    if not bvid or bvid in collab_bvids_all:
                        continue
                    if bvid in existing_bvids_this_mid:
                        stat["baseline_skip_existing"] += 1
                        continue

                    base_row = {
                        "project": BASELINE_PROJECT,
                        "bvid": bvid,
                        "url": f"https://www.bilibili.com/video/{bvid}",
                        "title": v.get("title",""),
                        "pubdate": v.get("pubdate", pd.NaT),
                        "owner_mid": mid,
                        "owner_name": disp,
                        "view": _safe_int(v.get("view",0)),
                        "reply": _safe_int(v.get("reply",0)),
                        "like": _safe_int(v.get("like",0)),
                        "coin": _safe_int(v.get("coin",0)),
                        "favorite": _safe_int(v.get("favorite",0)),
                        "danmaku": _safe_int(v.get("danmaku",0)),
                        "share": _safe_int(v.get("share",0)),
                        "fans_delta": 0,
                        "baseline_for": disp,
                        "data_type": "baseline",
                        "fetched_at": pd.Timestamp.now(),
                    }
                    baseline_rows_to_write[(BASELINE_PROJECT, bvid)] = base_row
                    existed_baseline.add(bvid)
                    stat["vlist_added"] += 1

                    detail = fetch_video_detail_by_bvid(bvid, sess=detail_sess, cookie=bili_cookie, proxy=bili_proxy)
                    if detail is not None and _norm_mid(detail.get("owner_mid", "")) == _norm_mid(mid):
                        detail_row = _detail_row_for_project(
                            detail,
                            project=BASELINE_PROJECT,
                            url=f"https://www.bilibili.com/video/{bvid}",
                            data_type="baseline",
                            baseline_for=disp,
                        )
                        # 详情接口的 owner_name 更准，但如果详情没有昵称，则保留合作库昵称
                        if not detail_row.get("owner_name"):
                            detail_row["owner_name"] = disp
                        baseline_rows_to_write[(BASELINE_PROJECT, bvid)] = detail_row
                        stat["detail_ok"] += 1
                    else:
                        stat["detail_fail"] += 1

                    _sleep_jitter(float(sleep_sec))

        # ========= A2) 写入 baseline =========
        if baseline_rows_to_write:
            df_new = normalize_df(pd.DataFrame(list(baseline_rows_to_write.values())))
            df_new["pubdate"] = pd.to_datetime(df_new["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df_new["fetched_at"] = pd.to_datetime(df_new["fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(df_new)

        progress.progress(1.0)
        status.success("KOL补齐流程完成。")

        if debug_rows and show_kol_quality_hint:
            with st.expander("本次KOL抓取诊断日志", expanded=True):
                dbg = pd.DataFrame(debug_rows)
                st.dataframe(dbg, use_container_width=True, height=320)

        if baseline_rows_to_write or rows_to_write:
            st.success(
                f"完成：合作BV详情刷新 {stat['collab_repair_ok']}/{stat['collab_repair_total']}；"
                f"本次处理不足KOL {stat['mid_selected']}/{stat['mid_need_total']}，跳过已足量 {stat['mid_skipped_enough']}；"
                f"新增 {stat['vlist_added']} 条基准；"
                f"跳过已存在 {stat['baseline_skip_existing']} 条；"
                f"列表失败 {stat['list_fail']}，列表空 {stat['list_empty']}；"
                f"详情补全成功 {stat['detail_ok']}，失败 {stat['detail_fail']}。"
                f"（数据已自动备份到 backup/backup_latest.csv）"
            )
            st.rerun()
        else:
            st.warning(
                f"本次未写入任何新数据：处理不足KOL {stat['mid_selected']}/{stat['mid_need_total']}，"
                f"跳过已足量 {stat['mid_skipped_enough']}，列表空 {stat['list_empty']}。"
                "请打开诊断日志看具体接口返回，必要时填入 B站 Cookie 后重试。"
            )

    st.markdown("**KOL基准诊断（按owner_mid统计库内数量）**")
    diag = []
    for mid in sorted(valid_mid_df["owner_mid"].unique().tolist()):
        owner_all = df_db[df_db["owner_mid"].apply(_norm_mid) == mid].copy()
        owner_all = _sort_owner_hist(owner_all)

        collab_bvids_this_mid = set(
            collab_df[collab_df["owner_mid"].apply(_norm_mid) == mid]["bvid"].astype(str).tolist()
        )
        base_pool = owner_all[
            (owner_all["project"] == BASELINE_PROJECT) |
            (~owner_all["project"].isin(set(active_projects)))
        ].copy()
        base_pool = base_pool[~base_pool["bvid"].astype(str).isin(collab_bvids_this_mid)]
        base_pool = base_pool[base_pool["view"].astype(float) > 0]
        base_pool = base_pool.drop_duplicates(subset=["bvid"], keep="last")
        base_pool = _sort_owner_hist(base_pool)

        avail = int(min(len(base_pool), baseline_window_n))
        if avail >= baseline_min_n:
            status_text = "OK"
            advice = "个人历史足量"
        elif avail > 0:
            status_text = f"小样本({avail}/{baseline_min_n})"
            advice = "可参考，建议继续补齐"
        else:
            status_text = "无个人历史基准"
            advice = "优先补齐：投稿/合集/动态/搜索/合作视频反查；多次为0时建议填Cookie"
        diag.append({
            "owner_mid": mid,
            "KOL/UP主": name_map.get(mid, owner_all["owner_name"].dropna().iloc[0] if not owner_all.empty else ""),
            "库内视频总数": int(len(owner_all)),
            "可用个人基准数": int(len(base_pool)),
            f"取最近{baseline_window_n}可用": avail,
            "状态": status_text,
            "补齐优先级": 0 if avail == 0 else (1 if avail < baseline_min_n else 9),
            "建议": advice,
        })
    if diag:
        diag_df = pd.DataFrame(diag).sort_values(["补齐优先级", "可用个人基准数"], ascending=[True, True])
        st.dataframe(diag_df, use_container_width=True, height=360)
    else:
        st.info("暂无可诊断的 owner_mid。点击“一键补齐”会先尝试用合作BV详情修复 owner_mid。")

    if btn_build_kol:
        df_all_m = normalize_df(load_all_rows())
        analysis_projects = active_projects if kol_view_scope == "当前评估项目" else list(collab_projects)
        baseline_exclude_projects = active_projects if kol_view_scope == "当前评估项目" else list(collab_projects)
        analysis_df_for_name = df_db[df_db["project"].isin(analysis_projects)].copy()
        analysis_df_for_name["owner_mid"] = analysis_df_for_name["owner_mid"].apply(_norm_mid)
        analysis_name_map = (
            analysis_df_for_name[analysis_df_for_name["owner_mid"].astype(str).str.len() > 0]
            .groupby("owner_mid")["owner_name"]
            .agg(lambda s: s.value_counts().index[0])
            .to_dict()
        ) if not analysis_df_for_name.empty else name_map
        lib = _build_kol_compare_lib(
            df_all_m=df_all_m,
            collab_projects=analysis_projects,
            baseline_window_n=baseline_window_n,
            baseline_min_n=baseline_min_n,
            name_map=analysis_name_map,
            use_proxy_baseline=bool(use_proxy_baseline),
            baseline_exclude_projects=baseline_exclude_projects,
        )
        if lib.empty:
            st.warning("没有生成KOL结果：请先补齐基准，或检查合作项目是否包含有效 owner_mid。")
        else:
            st.caption(f"当前KOL对比表统计范围：{kol_view_scope}｜项目数：{len(analysis_projects)}｜KOL数：{len(lib)}")
            tab_visual, tab_table = st.tabs(["📍 KOL视觉总览", "📋 KOL对比表校对"])
            with tab_visual:
                _render_kol_visuals(lib)
            with tab_table:
                display_cols = [
                    "owner_mid", "KOL/UP主", "推荐等级", "综合评分", "标注", "合作视频数", "基准样本数", "基准类型", "基准可靠性", "象限",
                    "标签", "KOL画像一句话", "合作播放中位数", "基准播放中位数", "播放提升",
                    "合作互动率中位数", "基准互动率中位数", "互动率提升",
                    "合作深度信号中位数", "基准深度信号中位数", "深度信号提升"
                ]
                show_lib = lib.copy()
                if "综合评分" in show_lib.columns:
                    show_lib["综合评分"] = pd.to_numeric(show_lib["综合评分"], errors="coerce").round(1)
                st.dataframe(show_lib[[c for c in display_cols if c in show_lib.columns]], use_container_width=True, height=560)
                st.download_button(
                    "⬇️ 下载KOL对比表（CSV）",
                    data=show_lib.to_csv(index=False).encode("utf-8-sig"),
                    file_name="kol_compare.csv",
                    mime="text/csv"
                )

else:
    st.info("请先在 KOL模块设置 中选择合作项目。")


# =========================================================
# BD decision center（新增：服务立项、资源分配、UP主选择）
# =========================================================
def _rank_pct(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(0)
    if len(x) <= 1:
        return pd.Series([70.0] * len(x), index=s.index)
    asc = not higher_is_better
    return x.rank(pct=True, ascending=asc).fillna(0.5) * 100


def _project_grade(score: float) -> str:
    score = _safe_float(score, 0)
    if score >= 82:
        return "S 可放量"
    if score >= 68:
        return "A 稳定继续"
    if score >= 54:
        return "B 小规模验证"
    if score >= 40:
        return "C 复盘优化"
    return "D 暂停谨慎"


def _project_action(grade: str) -> str:
    if grade.startswith("S"):
        return "优先排期，增加预算/资源位，复制高表现UP与内容结构"
    if grade.startswith("A"):
        return "保持合作规模，优先续约表现稳定UP"
    if grade.startswith("B"):
        return "小额测试，控制样本，验证题材与UP匹配"
    if grade.startswith("C"):
        return "先复盘素材/选题/UP匹配，再决定是否继续"
    return "暂停放量，仅保留低成本验证或更换合作方向"


def _kol_fit_tags(row: pd.Series) -> str:
    tags = []
    v = _safe_float(row.get("播放提升值"), np.nan)
    e = _safe_float(row.get("互动率提升值"), np.nan)
    d = _safe_float(row.get("深度信号提升值"), np.nan)
    score = _safe_float(row.get("综合评分"), np.nan)
    grade = str(row.get("推荐等级", ""))
    reliability = str(row.get("基准可靠性", ""))

    if not np.isnan(v) and v >= 0.35:
        tags.append("流量放大型")
    if not np.isnan(e) and e >= 0.20:
        tags.append("高互动型")
    if not np.isnan(d) and d >= 0.15:
        tags.append("深度沉淀型")
    if not np.isnan(score) and score >= 60 and reliability in ["高", "中"]:
        tags.append("稳定交付型")
    if not np.isnan(v) and v >= 0.35 and not np.isnan(e) and e < -0.15:
        tags.append("爆发高风险")
    if grade.startswith("D") or (not np.isnan(e) and e <= -0.30):
        tags.append("当前项目不适配")
    if not tags:
        tags.append("常规观察型")
    return "、".join(tags)


def _build_bd_kol_lib(df_all: pd.DataFrame, bd_projects: list[str]) -> pd.DataFrame:
    name_map_bd = {}
    d = df_all[df_all["project"].isin(bd_projects)].copy()
    if not d.empty:
        d["owner_mid"] = d["owner_mid"].apply(_norm_mid)
        name_map_bd = d.groupby("owner_mid")["owner_name"].agg(lambda s: s.value_counts().index[0]).to_dict()
    lib = _build_kol_compare_lib(
        df_all_m=df_all,
        collab_projects=bd_projects,
        baseline_window_n=baseline_window_n,
        baseline_min_n=baseline_min_n,
        name_map=name_map_bd,
        use_proxy_baseline=True,
    )
    if lib is None or lib.empty:
        return pd.DataFrame()
    lib = lib.copy()
    lib["推荐等级"] = lib["推荐等级"].astype(str)
    lib["适配标签"] = lib.apply(_kol_fit_tags, axis=1)
    return lib


def _build_project_decision_table(df_all: pd.DataFrame, bd_projects: list[str], kol_lib: pd.DataFrame) -> pd.DataFrame:
    d = df_all[(df_all["project"].isin(bd_projects)) & (df_all["project"] != BASELINE_PROJECT)].copy()
    if d.empty:
        return pd.DataFrame()
    d = compute_metrics(normalize_df(d))
    overall_er = float(d["engagement_rate"].median()) if not d.empty else 0
    overall_deep = float(d["deep_signal_ratio"].median()) if not d.empty else 0
    rows = []
    for proj, g in d.groupby("project"):
        g = g.sort_values("view", ascending=False).copy()
        total_view = int(g["view"].sum())
        video_cnt = int(len(g))
        up_cnt = int(g["owner_mid"].apply(_norm_mid).replace("", np.nan).nunique())
        er_med = float(g["engagement_rate"].median())
        deep_med = float(g["deep_signal_ratio"].median())
        top1_share = float(g.iloc[0]["view"] / total_view) if total_view > 0 and not g.empty else 0
        top3_share = float(g.head(3)["view"].sum() / total_view) if total_view > 0 else 0
        mids = set(g["owner_mid"].apply(_norm_mid).tolist())

        ksub = kol_lib[kol_lib["owner_mid"].astype(str).isin(mids)].copy() if kol_lib is not None and not kol_lib.empty else pd.DataFrame()
        kol_cnt = int(len(ksub))
        ab_cnt = int(ksub["推荐等级"].isin(["A 重点续约", "B 可继续"]).sum()) if not ksub.empty else 0
        de_cnt = int(ksub["推荐等级"].isin(["D 谨慎投放", "E 无法判断"]).sum()) if not ksub.empty else 0
        low_reliability = int(ksub["基准可靠性"].isin(["低", "无"]).sum()) if not ksub.empty and "基准可靠性" in ksub.columns else 0
        ab_rate = ab_cnt / kol_cnt if kol_cnt > 0 else 0
        de_rate = de_cnt / kol_cnt if kol_cnt > 0 else 0

        risks = []
        if video_cnt < 3:
            risks.append("样本不足")
        if top1_share >= 0.55 or top3_share >= 0.82:
            risks.append("单点依赖")
        if er_med < overall_er * 0.75 and total_view > d["view"].median() * max(3, video_cnt):
            risks.append("高播放低互动")
        if de_rate >= 0.35:
            risks.append("KOL匹配风险")
        if low_reliability >= max(1, kol_cnt * 0.35):
            risks.append("基准可靠性风险")
        if not risks:
            risks.append("暂无明显风险")

        rows.append({
            "project": proj,
            "视频数": video_cnt,
            "UP数": up_cnt,
            "总播放": total_view,
            "播放中位数": int(g["view"].median()) if video_cnt else 0,
            "互动率中位数": er_med,
            "深度信号中位数": deep_med,
            "Top1贡献": top1_share,
            "Top3贡献": top3_share,
            "KOL数": kol_cnt,
            "A/B可继续数": ab_cnt,
            "D/E风险数": de_cnt,
            "A/B占比": ab_rate,
            "风险标签": "、".join(risks),
            "风险数": 0 if risks == ["暂无明显风险"] else len(risks),
        })
    out = pd.DataFrame(rows)
    out["播放规模分"] = _rank_pct(out["总播放"], True)
    out["互动质量分"] = _rank_pct(out["互动率中位数"], True)
    out["沉淀质量分"] = _rank_pct(out["深度信号中位数"], True)
    out["KOL适配分"] = out["A/B占比"].fillna(0) * 100
    out["结构健康分"] = (100 - (out["Top1贡献"] * 55 + out["Top3贡献"] * 30 + out["风险数"] * 8)).clip(0, 100)
    out["决策分"] = (
        out["播放规模分"] * 0.25 +
        out["互动质量分"] * 0.25 +
        out["沉淀质量分"] * 0.18 +
        out["KOL适配分"] * 0.20 +
        out["结构健康分"] * 0.12
    ).round(1)
    out["项目分层"] = out["决策分"].apply(_project_grade)
    out["资源建议"] = out["项目分层"].apply(_project_action)
    return out.sort_values(["决策分", "总播放"], ascending=[False, False])


def _build_snapshot_compare(df_current: pd.DataFrame, df_snap: pd.DataFrame, bd_projects: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cur = df_current[(df_current["project"].isin(bd_projects)) & (df_current["project"] != BASELINE_PROJECT)].copy()
    cur = compute_metrics(normalize_df(cur)) if not cur.empty else cur
    if cur.empty:
        return pd.DataFrame(), pd.DataFrame()
    if df_snap is None or df_snap.empty:
        video_rows = cur.copy()
        video_rows["首日播放"] = np.nan
        video_rows["3日播放"] = np.nan
        video_rows["7日播放"] = np.nan
        video_rows["首日捕捉"] = "无"
    else:
        snap = df_snap[(df_snap["project"].isin(bd_projects)) & (df_snap["project"] != BASELINE_PROJECT)].copy()
        snap["pubdate"] = pd.to_datetime(snap["pubdate"], errors="coerce")
        snap["fetched_at"] = pd.to_datetime(snap["fetched_at"], errors="coerce")
        first_rows = []
        for (proj, bvid), sg in snap.groupby(["project", "bvid"]):
            sg = sg.sort_values("fetched_at")
            pub = sg["pubdate"].dropna().iloc[0] if not sg["pubdate"].dropna().empty else pd.NaT
            def max_until(days):
                if pd.isna(pub):
                    return np.nan
                pool = sg[(sg["fetched_at"] >= pub) & (sg["fetched_at"] <= pub + pd.Timedelta(days=days))]
                return float(pool["view"].max()) if not pool.empty else np.nan
            first_rows.append({
                "project": proj,
                "bvid": bvid,
                "首日播放": max_until(1),
                "3日播放": max_until(3),
                "7日播放": max_until(7),
                "快照数": int(len(sg)),
            })
        snap_agg = pd.DataFrame(first_rows)
        video_rows = cur.merge(snap_agg, on=["project", "bvid"], how="left") if not snap_agg.empty else cur.copy()
        for c in ["首日播放", "3日播放", "7日播放"]:
            if c not in video_rows.columns:
                video_rows[c] = np.nan
        video_rows["首日捕捉"] = np.where(pd.notna(video_rows["首日播放"]), "有", "无")

    video_rows["累计播放"] = video_rows["view"].astype(float)
    video_rows["累计/首日倍数"] = np.where(video_rows["首日播放"] > 0, video_rows["累计播放"] / video_rows["首日播放"], np.nan)
    video_rows["首日占累计"] = np.where(video_rows["累计播放"] > 0, video_rows["首日播放"] / video_rows["累计播放"], np.nan)

    def growth_type(r):
        if pd.isna(r.get("首日播放")):
            return "暂无首日快照"
        share = _safe_float(r.get("首日占累计"), np.nan)
        if not np.isnan(share) and share >= 0.70:
            return "首日爆发型"
        if not np.isnan(share) and share <= 0.35:
            return "长尾增长型"
        return "均衡增长型"
    video_rows["增长类型"] = video_rows.apply(growth_type, axis=1)

    proj_rows = []
    for proj, g in video_rows.groupby("project"):
        total_view = float(g["累计播放"].sum())
        first_sum = float(g["首日播放"].dropna().sum()) if g["首日播放"].notna().any() else np.nan
        capture_rate = float((g["首日捕捉"] == "有").mean()) if len(g) else 0
        proj_rows.append({
            "project": proj,
            "视频数": int(len(g)),
            "累计播放": int(total_view),
            "首日播放合计": int(first_sum) if not np.isnan(first_sum) else 0,
            "首日捕捉率": capture_rate,
            "累计/首日倍数": (total_view / first_sum) if first_sum and first_sum > 0 else np.nan,
            "主增长类型": g["增长类型"].value_counts().index[0] if not g.empty else "",
        })
    return video_rows, pd.DataFrame(proj_rows).sort_values("累计播放", ascending=False)


def _snapshot_has_within(sg: pd.DataFrame, pubdate, days: int) -> bool:
    if pd.isna(pubdate) or sg is None or sg.empty:
        return False
    s = sg.copy()
    s["fetched_at"] = pd.to_datetime(s["fetched_at"], errors="coerce")
    pool = s[(s["fetched_at"] >= pubdate) & (s["fetched_at"] <= pubdate + pd.Timedelta(days=days))]
    return not pool.empty


def _build_snapshot_health(df_current: pd.DataFrame, df_snap: pd.DataFrame, bd_projects: list[str]) -> pd.DataFrame:
    cur = df_current[(df_current["project"].isin(bd_projects)) & (df_current["project"] != BASELINE_PROJECT)].copy()
    if cur.empty:
        return pd.DataFrame()
    cur = normalize_df(cur)
    cur["pubdate"] = pd.to_datetime(cur["pubdate"], errors="coerce")
    now = pd.Timestamp.now()
    snap = df_snap.copy() if df_snap is not None and not df_snap.empty else pd.DataFrame()
    if not snap.empty:
        snap = snap[(snap["project"].isin(bd_projects)) & (snap["project"] != BASELINE_PROJECT)].copy()
        snap["fetched_at"] = pd.to_datetime(snap["fetched_at"], errors="coerce")

    rows = []
    for _, r in cur.iterrows():
        pub = r.get("pubdate", pd.NaT)
        age_hours = ((now - pub).total_seconds() / 3600.0) if pd.notna(pub) else np.nan
        sg = snap[(snap["project"] == r.get("project")) & (snap["bvid"] == r.get("bvid"))].copy() if not snap.empty else pd.DataFrame()
        has_1d = _snapshot_has_within(sg, pub, 1)
        has_3d = _snapshot_has_within(sg, pub, 3)
        has_7d = _snapshot_has_within(sg, pub, 7)
        if pd.isna(age_hours):
            status = "发布时间缺失"
        elif age_hours <= 24 and not has_1d:
            status = "首日窗口内，需尽快快照"
        elif age_hours > 24 and not has_1d:
            status = "首日已错过"
        elif age_hours <= 72 and not has_3d:
            status = "3日窗口内，需继续快照"
        elif age_hours > 72 and not has_3d:
            status = "3日已错过"
        elif age_hours <= 168 and not has_7d:
            status = "7日窗口内，需继续快照"
        elif age_hours > 168 and not has_7d:
            status = "7日已错过"
        else:
            status = "快照完整"
        rows.append({
            "project": r.get("project", ""),
            "bvid": r.get("bvid", ""),
            "title": r.get("title", ""),
            "owner_name": r.get("owner_name", ""),
            "pubdate": pub,
            "发布后小时": round(age_hours, 1) if not pd.isna(age_hours) else np.nan,
            "快照数": int(len(sg)),
            "首日快照": "有" if has_1d else "无",
            "3日快照": "有" if has_3d else "无",
            "7日快照": "有" if has_7d else "无",
            "快照状态": status,
        })
    out = pd.DataFrame(rows)
    order = {
        "首日窗口内，需尽快快照": 0,
        "3日窗口内，需继续快照": 1,
        "7日窗口内，需继续快照": 2,
        "首日已错过": 3,
        "3日已错过": 4,
        "7日已错过": 5,
        "发布时间缺失": 6,
        "快照完整": 9,
    }
    out["状态优先级"] = out["快照状态"].map(order).fillna(8)
    return out.sort_values(["状态优先级", "发布后小时"], ascending=[True, True])


def _restore_snapshot_backup(df_restore: pd.DataFrame) -> tuple[bool, int, str]:
    if df_restore is None or df_restore.empty:
        return False, 0, "文件为空"
    init_db()
    cols = [
        "project", "bvid", "snapshot_date", "fetched_at", "pubdate", "owner_mid", "owner_name", "title",
        "view", "like", "coin", "favorite", "reply", "danmaku", "share",
        "engagement", "engagement_rate", "deep_signal_ratio", "data_type"
    ]
    d = df_restore.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = None
    if "snapshot_date" in d.columns:
        d["snapshot_date"] = d["snapshot_date"].fillna("")
    missing_date = d["snapshot_date"].astype(str).str.strip() == ""
    d.loc[missing_date, "snapshot_date"] = pd.to_datetime(d.loc[missing_date, "fetched_at"], errors="coerce").dt.strftime("%Y-%m-%d")
    d["fetched_at"] = pd.to_datetime(d["fetched_at"], errors="coerce").fillna(pd.Timestamp.now()).dt.strftime("%Y-%m-%d %H:%M:%S")
    d["pubdate"] = pd.to_datetime(d["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    d = d[d["bvid"].astype(str).str.startswith("BV")].copy()
    if d.empty:
        return False, 0, "没有有效BV快照行"
    records = []
    for _, r in d[cols].iterrows():
        records.append(tuple(None if pd.isna(v) else v for v in r.tolist()))
    placeholders = ",".join(["?"] * len(cols))
    colnames = ",".join(cols)
    sql = f"INSERT OR REPLACE INTO {SNAPSHOT_TABLE_NAME} ({colnames}) VALUES ({placeholders})"
    with db_conn() as conn:
        conn.executemany(sql, records)
        conn.commit()
    try:
        _save_snapshot_backup_csv(load_snapshots())
    except Exception:
        pass
    return True, len(records), "恢复完成"


def _project_review_text(proj: str, decision_df: pd.DataFrame, kol_lib: pd.DataFrame, video_df: pd.DataFrame) -> str:
    if decision_df is None or decision_df.empty or proj not in decision_df["project"].astype(str).tolist():
        return "暂无可生成的复盘内容。"
    row = decision_df[decision_df["project"] == proj].iloc[0]
    vsub = video_df[video_df["project"] == proj].copy() if video_df is not None and not video_df.empty else pd.DataFrame()
    top_video = vsub.sort_values("累计播放", ascending=False).head(1).iloc[0] if not vsub.empty else None
    mids = set(vsub["owner_mid"].apply(_norm_mid).tolist()) if not vsub.empty and "owner_mid" in vsub.columns else set()
    ksub = kol_lib[kol_lib["owner_mid"].astype(str).isin(mids)].copy() if kol_lib is not None and not kol_lib.empty else pd.DataFrame()
    good = ksub[ksub["推荐等级"].isin(["A 重点续约", "B 可继续"])].sort_values("综合评分", ascending=False)["KOL/UP主"].head(5).tolist() if not ksub.empty else []
    risk = ksub[ksub["推荐等级"].isin(["D 谨慎投放", "E 无法判断"])].sort_values("综合评分", ascending=True)["KOL/UP主"].head(5).tolist() if not ksub.empty else []
    top_line = f"最高播放视频为《{top_video['title']}》，累计 {int(top_video['累计播放']):,} 播放，增长类型：{top_video['增长类型']}。" if top_video is not None else "暂无视频明细。"
    return "\n".join([
        f"项目【{proj}】当前分层为 {row['项目分层']}，决策分 {row['决策分']}，建议：{row['资源建议']}。",
        f"核心数据：{int(row['视频数'])} 条视频 / {int(row['UP数'])} 位UP，累计播放 {int(row['总播放']):,}，互动率中位数 {row['互动率中位数']*100:.2f}%，深度信号中位数 {row['深度信号中位数']*100:.1f}%。",
        top_line,
        f"推荐优先合作UP：{'、'.join(good) if good else '暂无明确A/B名单'}。",
        f"谨慎或需补充判断UP：{'、'.join(risk) if risk else '暂无明显风险名单'}。",
        f"风险提示：{row['风险标签']}。",
        "下一步动作：优先复用高表现UP与内容结构；对风险UP降低预算或更换选题；继续积累首日/3日/7日快照，用于判断内容起量速度与长尾能力。",
    ])


BD_GRADE_ORDER = ["S 可放量", "A 稳定继续", "B 小规模验证", "C 复盘优化", "D 暂停谨慎"]
BD_GRADE_COLORS = {
    "S 可放量": "#009E73",
    "A 稳定继续": "#0072B2",
    "B 小规模验证": "#E69F00",
    "C 复盘优化": "#D55E00",
    "D 暂停谨慎": "#7F7F7F",
}
BD_RISK_LABELS = ["样本不足", "单点依赖", "高播放低互动", "KOL匹配风险", "基准可靠性风险"]


def _split_tag_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame(columns=["标签", "数量"])
    for v in df[col].dropna().astype(str).tolist():
        for tag in [x.strip() for x in v.split("、") if x.strip()]:
            rows.append(tag)
    if not rows:
        return pd.DataFrame(columns=["标签", "数量"])
    return pd.Series(rows).value_counts().rename_axis("标签").reset_index(name="数量")


def _render_bd_visuals(decision_df: pd.DataFrame, kol_lib: pd.DataFrame, video_growth: pd.DataFrame, project_growth: pd.DataFrame):
    if decision_df is None or decision_df.empty:
        st.info("暂无可视化数据。")
        return

    d = decision_df.copy()
    d["项目分层"] = pd.Categorical(d["项目分层"].astype(str), categories=BD_GRADE_ORDER, ordered=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("项目数", f"{len(d):,}")
    c2.metric("S/A项目", f"{int(d['项目分层'].astype(str).isin(['S 可放量', 'A 稳定继续']).sum()):,}")
    c3.metric("风险项目", f"{int((d['风险数'] > 0).sum()):,}")
    c4.metric("总播放", f"{int(d['总播放'].sum()):,}")

    st.markdown("**项目决策矩阵（X=互动质量，Y=KOL适配，气泡=总播放）**")
    fig_matrix = px.scatter(
        d,
        x="互动质量分",
        y="KOL适配分",
        size="总播放",
        color="项目分层",
        color_discrete_map=BD_GRADE_COLORS,
        category_orders={"项目分层": BD_GRADE_ORDER},
        text="project",
        hover_data={
            "决策分": ":.1f",
            "总播放": True,
            "风险标签": True,
            "Top1贡献": ":.1%",
            "Top3贡献": ":.1%",
            "互动质量分": ":.1f",
            "KOL适配分": ":.1f",
        },
        height=560,
    )
    fig_matrix.add_vline(x=float(d["互动质量分"].median()), line_dash="dash", line_color="#666")
    fig_matrix.add_hline(y=float(d["KOL适配分"].median()), line_dash="dash", line_color="#666")
    fig_matrix.update_traces(textposition="top center", marker=dict(opacity=0.78, line=dict(width=1, color="white")))
    fig_matrix.update_layout(xaxis_title="互动质量分", yaxis_title="KOL适配分", legend_title_text="项目分层")
    st.plotly_chart(fig_matrix, use_container_width=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**项目决策分排行**")
        rank = d.sort_values("决策分", ascending=True)
        fig_rank = px.bar(
            rank,
            x="决策分",
            y="project",
            orientation="h",
            color="项目分层",
            color_discrete_map=BD_GRADE_COLORS,
            category_orders={"项目分层": BD_GRADE_ORDER},
            text="决策分",
            hover_data=["资源建议", "风险标签", "总播放"],
            height=max(380, min(760, 34 * len(rank) + 120)),
        )
        fig_rank.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_rank.update_layout(xaxis_range=[0, 105], yaxis_title="项目", legend_title_text="项目分层")
        st.plotly_chart(fig_rank, use_container_width=True)

    with right:
        st.markdown("**项目分层占比**")
        grade_count = d["项目分层"].astype(str).value_counts().reindex(BD_GRADE_ORDER).dropna().reset_index()
        grade_count.columns = ["项目分层", "数量"]
        fig_pie = px.pie(
            grade_count,
            names="项目分层",
            values="数量",
            color="项目分层",
            color_discrete_map=BD_GRADE_COLORS,
            hole=0.46,
            height=380,
        )
        fig_pie.update_layout(legend_title_text="项目分层")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("**风险热力图（1=命中风险）**")
    risk_rows = []
    for _, r in d.iterrows():
        hit = set(str(r.get("风险标签", "")).split("、"))
        row = {"project": r["project"]}
        for label in BD_RISK_LABELS:
            row[label] = 1 if label in hit else 0
        risk_rows.append(row)
    risk_df = pd.DataFrame(risk_rows)
    if not risk_df.empty:
        z = risk_df[BD_RISK_LABELS].to_numpy()
        fig_risk = px.imshow(
            z,
            x=BD_RISK_LABELS,
            y=risk_df["project"],
            color_continuous_scale=["#F6F8FA", "#D55E00"],
            aspect="auto",
            height=max(320, min(720, 30 * len(risk_df) + 140)),
        )
        fig_risk.update_traces(text=z, texttemplate="%{text}")
        fig_risk.update_layout(coloraxis_showscale=False, xaxis_title="风险类型", yaxis_title="项目")
        st.plotly_chart(fig_risk, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**KOL适配标签分布**")
        tag_count = _split_tag_counts(kol_lib, "适配标签")
        if tag_count.empty:
            st.info("暂无KOL标签数据。")
        else:
            fig_tags = px.bar(
                tag_count.sort_values("数量", ascending=True),
                x="数量",
                y="标签",
                orientation="h",
                text="数量",
                height=max(360, min(680, 34 * len(tag_count) + 120)),
                color="标签",
            )
            fig_tags.update_traces(textposition="outside", showlegend=False)
            fig_tags.update_layout(yaxis_title="适配标签", xaxis_title="KOL数量")
            st.plotly_chart(fig_tags, use_container_width=True)

    with c_right:
        st.markdown("**首日/累计增长类型分布**")
        if video_growth is None or video_growth.empty or "增长类型" not in video_growth.columns:
            st.info("暂无增长类型数据。")
        else:
            grow_count = video_growth["增长类型"].astype(str).value_counts().reset_index()
            grow_count.columns = ["增长类型", "视频数"]
            fig_growth_bar = px.bar(
                grow_count,
                x="增长类型",
                y="视频数",
                color="增长类型",
                text="视频数",
                height=360,
            )
            fig_growth_bar.update_traces(textposition="outside", showlegend=False)
            fig_growth_bar.update_layout(xaxis_title="增长类型", yaxis_title="视频数")
            st.plotly_chart(fig_growth_bar, use_container_width=True)

    st.markdown("**视频增长散点（有首日快照时：X=首日播放，Y=累计播放）**")
    if video_growth is None or video_growth.empty:
        st.info("暂无视频增长数据。")
    else:
        vg = video_growth.copy()
        vg["首日播放"] = pd.to_numeric(vg.get("首日播放", np.nan), errors="coerce")
        vg["累计播放"] = pd.to_numeric(vg.get("累计播放", np.nan), errors="coerce")
        plot_vg = vg.dropna(subset=["首日播放", "累计播放"]).copy()
        plot_vg = plot_vg[(plot_vg["首日播放"] > 0) & (plot_vg["累计播放"] > 0)].copy()
        if plot_vg.empty:
            st.info("当前还没有首日快照。上线后持续采集几次，散点图会自动出现。")
        else:
            fig_video = px.scatter(
                plot_vg,
                x="首日播放",
                y="累计播放",
                color="增长类型",
                size="累计播放",
                hover_name="title",
                hover_data=["project", "owner_name", "bvid", "累计/首日倍数"],
                log_x=True,
                log_y=True,
                height=560,
            )
            fig_video.update_layout(xaxis_title="首日播放（log）", yaxis_title="累计播放（log）")
            st.plotly_chart(fig_video, use_container_width=True)


st.divider()
st.subheader("BD项目决策辅助中心")
bd_default_projects = collab_projects if "collab_projects" in globals() and collab_projects else (sel_projects if sel_projects else projects)
bd_projects = st.multiselect("BD决策分析项目范围", projects, default=bd_default_projects, key="bd_decision_projects")

if not bd_projects:
    st.info("请选择至少一个项目用于BD决策分析。")
else:
    with st.spinner("正在生成BD决策分析..."):
        bd_kol_lib = _build_bd_kol_lib(df_db, bd_projects)
        bd_project_decision = _build_project_decision_table(df_db, bd_projects, bd_kol_lib)
        bd_snapshots = load_snapshots()
        bd_video_growth, bd_project_growth = _build_snapshot_compare(df_db, bd_snapshots, bd_projects)

    tab_bd0, tab_bd1, tab_bd2, tab_bd3, tab_bd4, tab_bd5, tab_bd6 = st.tabs([
        "可视化总览", "项目分层", "KOL适配标签", "首日/累计表现", "风险状态", "复盘模板", "快照备份"
    ])

    with tab_bd0:
        _render_bd_visuals(bd_project_decision, bd_kol_lib, bd_video_growth, bd_project_growth)

    with tab_bd1:
        st.markdown("**项目分层总览（服务立项与资源分配）**")
        if bd_project_decision.empty:
            st.info("暂无项目分层数据。")
        else:
            show = bd_project_decision.copy()
            for c in ["互动率中位数", "深度信号中位数", "Top1贡献", "Top3贡献", "A/B占比"]:
                show[c] = (show[c] * 100).map(lambda x: f"{x:.1f}%")
            st.dataframe(show[[
                "project", "项目分层", "决策分", "资源建议", "视频数", "UP数", "总播放",
                "互动率中位数", "深度信号中位数", "Top1贡献", "Top3贡献", "A/B可继续数", "D/E风险数", "风险标签"
            ]], use_container_width=True, height=420)
            st.download_button(
                "下载项目分层CSV",
                data=bd_project_decision.to_csv(index=False).encode("utf-8-sig"),
                file_name="bd_project_decision.csv",
                mime="text/csv",
            )

    with tab_bd2:
        st.markdown("**KOL适配标签库（服务UP主选择）**")
        if bd_kol_lib.empty:
            st.info("暂无KOL适配标签。请先补齐KOL基准。")
        else:
            show = bd_kol_lib.copy()
            if "综合评分" in show.columns:
                show["综合评分"] = pd.to_numeric(show["综合评分"], errors="coerce").round(1)
            display = [
                "owner_mid", "KOL/UP主", "推荐等级", "综合评分", "适配标签", "基准类型", "基准可靠性",
                "播放提升", "互动率提升", "深度信号提升", "合作播放中位数", "基准播放中位数"
            ]
            st.dataframe(show[[c for c in display if c in show.columns]], use_container_width=True, height=520)
            st.download_button(
                "下载KOL适配标签CSV",
                data=show.to_csv(index=False).encode("utf-8-sig"),
                file_name="bd_kol_fit_tags.csv",
                mime="text/csv",
            )

    with tab_bd3:
        st.markdown("**首日 / 3日 / 7日 / 累计表现对比**")
        st.caption("说明：首日/3日/7日依赖快照表。从这版开始自动积累；过去没有快照的数据不会伪造。")
        if bd_project_growth.empty:
            st.info("暂无快照对比数据。当前版本上线后，每次采集/导入/补齐会自动记录快照。")
        else:
            show_proj = bd_project_growth.copy()
            show_proj["首日捕捉率"] = (show_proj["首日捕捉率"] * 100).map(lambda x: f"{x:.1f}%")
            show_proj["累计/首日倍数"] = show_proj["累计/首日倍数"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}x")
            st.dataframe(show_proj, use_container_width=True, height=260)
            show_video = bd_video_growth.copy()
            for c in ["首日播放", "3日播放", "7日播放", "累计播放"]:
                show_video[c] = pd.to_numeric(show_video[c], errors="coerce")
            show_cols = [
                "project", "bvid", "title", "owner_name", "pubdate", "首日捕捉",
                "首日播放", "3日播放", "7日播放", "累计播放", "累计/首日倍数", "增长类型"
            ]
            st.dataframe(show_video[[c for c in show_cols if c in show_video.columns]].sort_values("累计播放", ascending=False), use_container_width=True, height=480)

    with tab_bd4:
        st.markdown("**风险状态提示**")
        if bd_project_decision.empty:
            st.info("暂无风险数据。")
        else:
            risk_df = bd_project_decision[["project", "项目分层", "风险标签", "风险数", "Top1贡献", "Top3贡献", "D/E风险数", "资源建议"]].copy()
            risk_df["Top1贡献"] = (risk_df["Top1贡献"] * 100).map(lambda x: f"{x:.1f}%")
            risk_df["Top3贡献"] = (risk_df["Top3贡献"] * 100).map(lambda x: f"{x:.1f}%")
            st.dataframe(risk_df.sort_values(["风险数", "D/E风险数"], ascending=False), use_container_width=True, height=360)

    with tab_bd5:
        st.markdown("**一键复盘模板**")
        if bd_project_decision.empty:
            st.info("暂无可生成复盘的项目。")
        else:
            review_project = st.selectbox("选择要生成复盘的项目", bd_project_decision["project"].tolist(), key="bd_review_project")
            st.text_area(
                "复盘文案",
                value=_project_review_text(review_project, bd_project_decision, bd_kol_lib, bd_video_growth),
                height=260,
            )

    with tab_bd6:
        st.markdown("**快照备份 / 恢复 / 健康检查**")
        st.caption("首日、3日、7日表现依赖快照。BV可以重拉累计，但错过的历史快照无法准确补回。建议定期下载快照备份CSV。")

        snap_all = load_snapshots()
        health_df = _build_snapshot_health(df_db, snap_all, bd_projects)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("快照记录数", 0 if snap_all is None or snap_all.empty else len(snap_all))
        s2.metric("覆盖视频数", 0 if snap_all is None or snap_all.empty else snap_all["bvid"].nunique())
        if health_df.empty:
            s3.metric("首日覆盖率", "-")
            s4.metric("需关注视频", "-")
        else:
            s3.metric("首日覆盖率", f"{(health_df['首日快照'].eq('有').mean() * 100):.1f}%")
            s4.metric("需关注视频", int((health_df["快照状态"] != "快照完整").sum()))

        if snap_all is not None and not snap_all.empty:
            st.download_button(
                "下载快照备份CSV",
                data=snap_all.to_csv(index=False).encode("utf-8-sig"),
                file_name="backup_snapshots_latest.csv",
                mime="text/csv",
            )
        else:
            st.info("当前暂无快照可下载。页面打开后会自动记录当前项目数据为今日快照。")

        uploaded_snap = st.file_uploader("上传快照备份CSV恢复", type=["csv"], key="snapshot_restore_upload")
        if uploaded_snap is not None and st.button("恢复快照备份", key="snapshot_restore_btn"):
            raw = uploaded_snap.getvalue()
            df_restore = None
            for enc in ["utf-8-sig", "utf-8", "gbk"]:
                try:
                    df_restore = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    break
                except Exception:
                    df_restore = None
            ok, cnt, msg = _restore_snapshot_backup(df_restore)
            if ok:
                st.success(f"{msg}：{cnt} 条快照。")
                st.rerun()
            else:
                st.error(f"恢复失败：{msg}")

        if not health_df.empty:
            st.markdown("**快照健康检查**")
            status_counts = health_df["快照状态"].value_counts().reset_index()
            status_counts.columns = ["快照状态", "视频数"]
            fig_health = px.bar(
                status_counts,
                x="快照状态",
                y="视频数",
                color="快照状态",
                text="视频数",
                height=360,
            )
            fig_health.update_traces(textposition="outside", showlegend=False)
            fig_health.update_layout(xaxis_title="状态", yaxis_title="视频数")
            st.plotly_chart(fig_health, use_container_width=True)

            show_health = health_df.copy()
            show_health["pubdate"] = pd.to_datetime(show_health["pubdate"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(
                show_health[[
                    "project", "bvid", "title", "owner_name", "pubdate", "发布后小时",
                    "快照数", "首日快照", "3日快照", "7日快照", "快照状态"
                ]],
                use_container_width=True,
                height=460,
            )

            st.download_button(
                "下载BV清单+快照状态CSV",
                data=show_health.to_csv(index=False).encode("utf-8-sig"),
                file_name="snapshot_health_check.csv",
                mime="text/csv",
            )
