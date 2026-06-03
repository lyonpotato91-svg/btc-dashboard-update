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

st.set_page_config(page_title="B站运营数据Dashboard - 深度优化版", layout="wide")
_BOOT_NOTICE = st.empty()
_BOOT_NOTICE.info("应用正在启动，请稍等。如果长时间停在这里，通常是数据库/备份文件过大或云端正在重启。")

# =========================
# Constants
# =========================
BASELINE_PROJECT = "__BASELINE__"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "bili_dashboard.db")

BACKUP_DIR = os.path.join(APP_DIR, "backup")
BACKUP_LATEST_CSV = os.path.join(BACKUP_DIR, "backup_latest.csv")
BACKUP_SNAPSHOTS_CSV = os.path.join(BACKUP_DIR, "backup_snapshots_latest.csv")

TABLE_NAME = "videos"
SNAPSHOT_TABLE_NAME = "video_snapshots"

# =========================
# ✅ 深度优化：随机User-Agent池
# =========================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.119 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.86 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.95 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.119 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]

HEADERS_TEMPLATE = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://space.bilibili.com",
    "Referer": "https://space.bilibili.com/",
}

# =========================
# DB Functions
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
    if df_all is None or df_all.empty:
        return
    _ensure_backup_dir()
    try:
        df_all.to_csv(BACKUP_LATEST_CSV, index=False, encoding="utf-8-sig")
    except Exception:
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
    try:
        init_db()
        with db_conn() as conn:
            df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

        if df is not None and not df.empty:
            return df

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
# Performance labels
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

@st.cache_data(ttl=60*3)  # ✅ 深度优化：3分钟缓存，更频繁更新
def _get_wbi_keys() -> tuple[str, str]:
    nav = "https://api.bilibili.com/x/web-interface/nav"
    headers = HEADERS_TEMPLATE.copy()
    headers["User-Agent"] = random.choice(USER_AGENTS)
    r = requests.get(nav, headers=headers, timeout=10)
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
    query = urllib.parse.urlencode(sorted_items, quote_via=urllib.parse.quote)
    w_rid = hashlib.md5((query + mixin_key).encode("utf-8")).hexdigest()
    params["w_rid"] = w_rid
    return params

# =========================
# ✅ 深度优化：智能延迟系统
# =========================
class SmartDelayManager:
    """智能延迟管理器：根据请求情况动态调整延迟"""
    
    def __init__(self):
        self.request_count = 0
        self.last_failure_time = 0
        self.consecutive_failures = 0
        self.base_delay = 1.0
        self.max_delay = 5.0
        
    def get_delay(self) -> float:
        """计算当前应该使用的延迟时间"""
        now = time.time()
        
        # 基础延迟：1.0-2.0秒随机
        delay = self.base_delay + random.uniform(0.3, 1.0)
        
        # 每个请求后，基础延迟略微增加（模拟人类行为）
        self.request_count += 1
        if self.request_count % 5 == 0:
            # 每5个请求后，额外等待
            delay += random.uniform(1.0, 2.0)
        
        if self.request_count % 20 == 0:
            # 每20个请求后，长休息
            delay += random.uniform(3.0, 5.0)
        
        # 如果最近有失败，增加延迟
        if self.consecutive_failures > 0:
            delay = min(delay * (1.5 ** self.consecutive_failures), self.max_delay)
        
        # 如果距离上次失败时间很短，增加延迟
        if now - self.last_failure_time < 60:
            delay += random.uniform(0.5, 1.5)
        
        return delay
    
    def report_success(self):
        """报告成功请求"""
        self.consecutive_failures = 0
        
    def report_failure(self):
        """报告失败请求"""
        self.last_failure_time = time.time()
        self.consecutive_failures += 1
        
    def reset(self):
        """重置计数器"""
        self.request_count = 0
        self.consecutive_failures = 0

# 全局延迟管理器
_delay_manager = SmartDelayManager()

def _sleep_smart():
    """使用智能延迟系统"""
    delay = _delay_manager.get_delay()
    time.sleep(delay)

def _sleep_jitter(base: float = 1.0):
    """兼容旧版的延迟函数"""
    _sleep_smart()


# =========================
# ✅ 深度优化：增强的Session管理
# =========================
def _apply_cookie_to_session(sess: requests.Session, cookie: str = "") -> requests.Session:
    cookie = (cookie or "").strip()
    if cookie:
        sess.headers.update({"Cookie": cookie})
    return sess


def _normalize_proxy_url(proxy: str = "") -> str:
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
    """获取更完整的访客Cookie"""
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
        
        # 生成必要的cookie
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
    """建立更真实的Session，完整模拟浏览器访问路径"""
    sess = requests.Session()
    
    # ✅ 深度优化：随机选择User-Agent
    h = HEADERS_TEMPLATE.copy()
    h["User-Agent"] = random.choice(USER_AGENTS)
    
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
    
    # ✅ 深度优化：更完整的预热流程
    try:
        # 1. 先访问首页，建立基础cookie
        sess.get("https://www.bilibili.com/", timeout=10)
        time.sleep(random.uniform(0.2, 0.5))
        
        # 2. 获取访客指纹
        _seed_bili_guest_cookies(sess)
        time.sleep(random.uniform(0.1, 0.3))
        
        # 3. 访问目标空间页（如果referer包含space.bilibili.com）
        if "space.bilibili.com" in referer:
            try:
                sess.get(referer, timeout=10)
                time.sleep(random.uniform(0.1, 0.3))
            except Exception:
                pass
                
    except Exception:
        pass
        
    return sess


def _bili_num_to_int(x, default=0) -> int:
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
        "share": _bili_num_to_int(v.get("share", stat
