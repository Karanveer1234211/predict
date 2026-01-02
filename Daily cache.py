
#!/usr/bin/env python3
"""
New_Cache_daily_only.py
Daily-only cache builder for Zerodha Kite historical data.
Writes <SYMBOL>_daily.parquet with the same full daily indicators you already use.
Intraday/ORB logic is completely removed.
Based on your existing script's daily pipeline & indicators.

Changes in this version:
- Incremental caching: fetch only missing days after last cached date,
  merge with existing parquet, and recompute indicators for correctness.
- CLI flag --incremental to simplify daily tail updates.
"""

from __future__ import annotations
import concurrent.futures as cf
import contextlib
import datetime as dt
import functools
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict

import numpy as np
import pandas as pd

# ----------------------------- GUI (tkinter) -----------------------------
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
    from tkinter import ttk
    TK_OK = True
except Exception:
    TK_OK = False

# --------------------- Timezone / market session -------------------------
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
DEFAULT_SESSION_OPEN = dt.time(9, 15, tzinfo=IST)
DEFAULT_SESSION_CLOSE = dt.time(15, 30, tzinfo=IST)

def today_ist() -> dt.date:
    return dt.datetime.now(tz=IST).date()

# ----------------------------- Schema + .ok -------------------------------
SCHEMA_VERSION = 12  # keep parity with your existing daily schema
OK_VERSION_KEY = "schema_version"

# ------------------ Windows default cache roots ---------------------------
WIN_DEFAULT_BASE = Path(r"C:\Users\karanvsi\Desktop\Pycharm\Cache")

def _expand_path(value: str) -> Path:
    return Path(value).expanduser()

def _default_base_dir() -> Path:
    env_base = os.environ.get("CACHE_BASE_DIR")
    if env_base:
        return _expand_path(env_base)
    if os.name == "nt":
        return WIN_DEFAULT_BASE
    return Path.home() / ".kite_cache"

def _platform_default(env_var: str, *, windows_default: Path, unix_suffix: str) -> Path:
    val = os.environ.get(env_var)
    if val:
        return _expand_path(val)
    base = _default_base_dir()
    if os.name == "nt":
        return windows_default
    return base / unix_suffix

def _default_daily_root() -> Path:
    return _platform_default(
        "CACHE_DAILY_ROOT",
        windows_default=WIN_DEFAULT_BASE / "cache_daily_new",
        unix_suffix="cache_daily_new",
    )

# -------------------------------- Config ----------------------------------
@dataclass(frozen=True)
class Config:
    # Paths
    daily_root: Path = field(default_factory=_default_daily_root)

    # Session (unused directly for daily; kept for any future alignment)
    trading_open: dt.time = DEFAULT_SESSION_OPEN
    trading_close: dt.time = DEFAULT_SESSION_CLOSE

    # Execution defaults (global)
    max_workers: int = 24
    rate_limit_per_sec: float = 12.0
    request_timeout_s: float = 15.0
    retry_tries: int = 6
    retry_backoff_base: float = 0.45

    # FAST path
    parquet_engine: str = os.environ.get("PARQUET_ENGINE", "pyarrow")
    parquet_compression: Optional[str] = os.environ.get("PARQUET_COMPRESSION", None)
    parquet_use_dictionary: bool = False

    def day_root(self) -> Path:
        return self.daily_root

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        kwargs = {}
        env_key = "CACHE_DAILY_ROOT"
        val = os.environ.get(env_key)
        if val:
            kwargs["daily_root"] = _expand_path(val)
        kwargs.update(overrides)
        return cls(**kwargs)

    def with_updates(self, **updates) -> "Config":
        return replace(self, **updates)

# ------------------------- Sanitization / paths ---------------------------
_ILLEGAL = set('<>:"/\n?*')
_HEADER_WORDS = {"symbol", "symbols", "ticker", "tickers", "scrip", "scrips", "name"}

def sanitize_symbol(sym: str) -> Optional[str]:
    if sym is None:
        return None
    s = str(sym)
    s = s.replace("\x00", "").replace("\r", " ").replace("\t", " ").replace("\n", " ")
    s = s.lstrip("\ufeff")
    s = " ".join(s.strip().split())
    s = "".join(ch for ch in s if ch not in _ILLEGAL)
    if not s:
        return None
    if s.strip().casefold() in _HEADER_WORDS:
        return None
    return s

def assert_path_safe(p: Path):
    sp = str(p)
    if "\x00" in sp:
        raise ValueError(f"Path contains NUL (\\x00): {sp!r}")

def daily_path(config: Config, symbol: str) -> Path:
    s = sanitize_symbol(symbol) or "UNKNOWN"
    p = config.day_root() / f"{s}_daily.parquet"
    assert_path_safe(p)
    return p

def ok_path(config: Config, symbol: str) -> Path:
    s = sanitize_symbol(symbol) or "UNKNOWN"
    p = config.day_root() / f"{s}_daily.ok.json"
    assert_path_safe(p)
    return p

def ok_meta_base() -> dict:
    return {OK_VERSION_KEY: SCHEMA_VERSION, "created_ts": dt.datetime.now(tz=IST).isoformat()}

# ------------------------------- Atomic IO --------------------------------
class FileLock:
    def __init__(self, path: Path, poll_ms: int = 50, timeout_s: float = 30.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = Path(str(path) + ".lock")
        self.poll_ms = poll_ms
        self.timeout_s = timeout_s
        self._fd: Optional[int] = None

    def acquire(self):
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode())
                return
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(f"Timeout acquiring lock {self.lock_path}")
                time.sleep(self.poll_ms / 1000.0)

    def release(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.lock_path)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

def atomic_write_bytes(target: Path, data: bytes):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)

def write_json_atomic(path: Path, obj: dict):
    raw = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True).encode()
    atomic_write_bytes(path, raw)

def to_parquet(path: Path, df: pd.DataFrame, *, engine: str, compression: Optional[str], use_dictionary: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine=engine, compression=compression, use_dictionary=use_dictionary)

def read_parquet(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if columns is not None:
        columns = list(columns)
    return pd.read_parquet(path, columns=columns)

def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --------------------------- Rate limit + retry ---------------------------
class RateLimiter:
    def __init__(self, per_sec: float):
        self.per_sec = float(per_sec)
        self._lock = threading.Lock()
        self._tokens = per_sec
        self._updated = time.perf_counter()

    def acquire(self):
        while True:
            with self._lock:
                now = time.perf_counter()
                self._tokens = min(self.per_sec, self._tokens + (now - self._updated) * self.per_sec)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            need = max(0.0, 1.0 - self._tokens)
            wait = need / self.per_sec if self.per_sec > 0 else 0.0
            time.sleep(wait if wait > 0 else 0)

def with_retry(fn, *, tries: int, backoff: float):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        attempt = 0
        last_exc = None
        while attempt < tries:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e).lower()
                if "too many" in msg or "rate" in msg or "429" in msg:
                    sleep = min(15.0, backoff * (2.2 ** attempt))
                else:
                    sleep = min(8.0, backoff * (1.8 ** attempt) + np.random.random() * (backoff / 2))
                last_exc = e
                time.sleep(sleep)
                attempt += 1
        raise last_exc
    return wrapper

class DataFrameCache:
    def __init__(self, maxsize: int = 256):
        self.maxsize = max(1, int(maxsize))
        self._store: "dict[tuple, pd.DataFrame]" = {}
        self._order: List[tuple] = []
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[pd.DataFrame]:
        with self._lock:
            df = self._store.get(key)
            if df is None:
                return None
            # LRU bump
            if key in self._order:
                self._order.remove(key)
                self._order.append(key)
            return df.copy(deep=True)

    def put(self, key: tuple, df: pd.DataFrame) -> pd.DataFrame:
        clone = df.copy(deep=True)
        with self._lock:
            self._store[key] = clone
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            while len(self._order) > self.maxsize:
                old = self._order.pop(0)
                self._store.pop(old, None)
        return clone.copy(deep=True)

# ---------------------- Provider + resolver (Kite) ------------------------
try:
    from kiteconnect import KiteConnect
    from kiteconnect.exceptions import TokenException, KiteException, InputException
except Exception:
    KiteConnect = None
    TokenException = KiteException = InputException = Exception

class AuthExpired(Exception):
    """Kite access token missing/expired/invalid."""

def _token_file_path() -> str:
    env_path = os.environ.get("KITE_TOKEN_FILE")
    if env_path:
        return env_path
    default_win = r"C:\Users\karanvsi\PyCharmMiscProject\kite_token.json"
    if os.name == "nt" and os.path.exists(default_win):
        return default_win
    return os.path.join(os.path.dirname(__file__), "kite_token.json")

def _instrument_cache_path() -> Path:
    p = os.environ.get("INSTRUMENT_CACHE_FILE")
    if p:
        return Path(p)
    default_win = str(WIN_DEFAULT_BASE / "instrument_cache.json")
    return Path(default_win) if os.name == "nt" else Path("instrument_cache.json")

def _load_instrument_cache() -> dict:
    try:
        p = _instrument_cache_path()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_instrument_cache(cache: dict) -> None:
    try:
        p = _instrument_cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

import difflib
UNRESOLVED_LOG = Path(os.environ.get("UNRESOLVED_SYMBOLS_LOG") or str(_instrument_cache_path().parent / "unresolved_symbols.jsonl"))
OVERRIDES_FILE = Path(os.environ.get("SYMBOL_OVERRIDES_FILE") or str(_instrument_cache_path().parent / "symbol_overrides.json"))

class UnresolvedSymbol(Exception):
    """Symbol cannot be mapped to instrument_token."""

def _normalize_sym(s: str) -> str:
    s = (s or "").upper().strip()
    for suf in ("-EQ", "-BE", "-BZ", "-BL", "-SM", "-GS", "-GB"):
        if s.endswith(suf):
            s = s[:-len(suf)]
    return "".join(ch for ch in s if ch.isalnum())

def _load_overrides() -> dict:
    try:
        if OVERRIDES_FILE.exists():
            with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {_normalize_sym(k): str(v).upper() for k, v in d.items()}
    except Exception:
        pass
    return {}

def _append_unresolved_log(symbol: str, suggestions: list):
    val = os.environ.get("SKIP_UNRESOLVED", "").strip().lower()
    if val in ("1", "true", "yes"):
        return
    UNRESOLVED_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": dt.datetime.now(tz=IST).isoformat(),
        "symbol": symbol,
        "suggestions": suggestions[:5],
    }
    with open(UNRESOLVED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

class InvalidInstrument(Exception):
    """Instrument token invalid/stale."""

class SymbolResolver:
    def __init__(self, kite: KiteConnect):
        self.kite = kite
        self.overrides = _load_overrides()
        self._built = False
        self.exact = {}
        self.base = {}
        self.names = []

    def _build_maps(self):
        if self._built:
            return
        rows = self.kite.instruments("NSE") + self.kite.instruments("BSE")
        by_exact, by_base = {}, {}
        for r in rows:
            ts = str(r.get("tradingsymbol", "")).upper()
            itok = r.get("instrument_token")
            seg = str(r.get("segment", ""))
            inst_type = str(r.get("instrument_type", ""))
            if not itok or not ts:
                continue
            score = 2 if (seg.upper().startswith(("NSE", "BSE")) and inst_type.upper() == "EQ") else 1 if seg.upper().startswith(("NSE", "BSE")) else 0
            prev = by_exact.get(ts)
            if (prev is None) or (score > prev[0]):
                by_exact[ts] = (score, int(itok))
            base = _normalize_sym(ts)
            prevb = by_base.get(base)
            if (prevb is None) or (score > prevb[0]):
                by_base[base] = (score, int(itok))
        self.exact = {k: v[1] for k, v in by_exact.items()}
        self.base = {k: v[1] for k, v in by_base.items()}
        self.names = list(self.exact.keys())
        self._built = True

    def resolve(self, symbol: str) -> int | None:
        norm = _normalize_sym(symbol)
        if norm in self.overrides:
            want = self.overrides[norm]
            self._build_maps()
            tok = self.exact.get(want) or self.exact.get(f"{want}-EQ") or self.base.get(_normalize_sym(want))
            if tok:
                return int(tok)
        self._build_maps()
        if symbol.upper() in self.exact:
            return int(self.exact[symbol.upper()])
        if f"{symbol.upper()}-EQ" in self.exact:
            return int(self.exact[f"{symbol.upper()}-EQ"])
        if norm in self.base:
            return int(self.base[norm])
        close = difflib.get_close_matches(symbol.upper(), self.names, n=5, cutoff=0.77)
        _append_unresolved_log(symbol, close)
        return None

class KiteProvider:
    """Zerodha Kite-backed provider with symbol->instrument caching (daily-only)."""
    def __init__(self, *, exchange_prefix: str = "NSE:"):
        if KiteConnect is None:
            raise RuntimeError("kiteconnect not installed. `pip install kiteconnect`")
        self.exchange_prefix = exchange_prefix.rstrip(":") + ":"
        self._kite: Optional[KiteConnect] = None
        self._instruments: Dict[str, int] = {}
        self._inst_cache_file: Path = _instrument_cache_path()
        self._inst_cache_data: Dict[str, int] = {
            k.upper(): int(v)
            for k, v in _load_instrument_cache().items()
            if isinstance(v, (int, float, str)) and str(v).isdigit()
        }
        self._load_token()
        self._resolver = SymbolResolver(self._kite)

    def _load_token(self) -> None:
        token_file = _token_file_path()
        if not os.path.exists(token_file):
            raise AuthExpired(f"Token file missing: {token_file}. Refresh it first.")
        with open(token_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        api_key = data.get("api_key")
        access_token = data.get("access_token")
        if not api_key or not access_token:
            raise AuthExpired("api_key/access_token missing in token file.")
        kite = KiteConnect(api_key=api_key)
        try:
            kite.set_access_token(access_token)
            _ = kite.profile()
        except Exception as e:
            raise AuthExpired(str(e))
        self._kite = kite

    def _symbol_to_instrument_token(self, symbol: str) -> int:
        sym = symbol.strip().upper()
        if sym in self._instruments:
            return self._instruments[sym]
        if sym in self._inst_cache_data:
            tok = int(self._inst_cache_data[sym])
            self._instruments[sym] = tok
            return tok

        assert self._kite is not None
        # Try LTP quick path
        qual = f"{self.exchange_prefix}{sym}"
        try:
            quote = self._kite.ltp([qual])
            if quote and isinstance(quote, dict):
                info = quote.get(qual) or (list(quote.values())[0] if list(quote.values()) else None)
                if info and "instrument_token" in info:
                    inst = int(info["instrument_token"])
                    self._instruments[sym] = inst
                    self._inst_cache_data[sym] = inst
                    _save_instrument_cache(self._inst_cache_data)
                    return inst
        except (TokenException, InputException) as e:
            if "token" in str(e).lower():
                raise AuthExpired(str(e))
            raise
        except KiteException:
            pass

        # Fallback to instruments dump + resolver
        rows = self._kite.instruments("NSE")
        by_exact, by_base = {}, {}
        for r in rows:
            ts = str(r.get("tradingsymbol", "")).upper()
            itok = r.get("instrument_token")
            seg = str(r.get("segment", ""))
            inst_type = str(r.get("instrument_type", ""))
            if not itok or not ts:
                continue
            score = 2 if (seg.upper().startswith("NSE") and inst_type.upper() == "EQ") else 1 if seg.upper().startswith("NSE") else 0
            prev = by_exact.get(ts)
            if (prev is None) or (score > prev[0]):
                by_exact[ts] = (score, int(itok))
            base = ts[:-3] if ts.endswith("-EQ") else ts
            prevb = by_base.get(base)
            if (prevb is None) or (score > prevb[0]):
                by_base[base] = (score, int(itok))

        token = None
        if sym in by_exact:
            token = by_exact[sym][1]
        elif f"{sym}-EQ" in by_exact:
            token = by_exact[f"{sym}-EQ"][1]
        elif sym in by_base:
            token = by_base[sym][1]
        else:
            base = sym[:-3] if sym.endswith("-EQ") else sym
            token = by_exact.get(base, (None, None))[1] or by_base.get(base, (None, None))[1]
        if token is None:
            resolved = self._resolver.resolve(symbol)
            if resolved is not None:
                token = int(resolved)
        if token is None:
            names = list(by_exact.keys())
            close = difflib.get_close_matches(sym, names, n=3, cutoff=0.80)
            _append_unresolved_log(symbol, close)
            raise UnresolvedSymbol(sym)
        inst = int(token)
        self._instruments[sym] = inst
        self._inst_cache_data[sym] = inst
        _save_instrument_cache(self._inst_cache_data)
        return inst

    def _hist(self, instrument_token: int, start_dt: dt.datetime, end_dt: dt.datetime, interval: str):
        assert self._kite is not None
        try:
            return self._kite.historical_data(
                instrument_token,
                from_date=start_dt,
                to_date=end_dt,
                interval=interval,
                oi=False,
            )
        except (TokenException, InputException) as e:
            msg_low = str(e).lower()
            if "instrument_token" in msg_low:
                raise InvalidInstrument(str(e))
            if "access token" in msg_low or ("token" in msg_low and "instrument_token" not in msg_low):
                raise AuthExpired(str(e))
            if "too many" in msg_low or "429" in msg_low:
                raise
            raise RuntimeError(f"Kite historical data failed: {e}")

    def _ensure_ist_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "date" in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            try:
                import zoneinfo
                tz = zoneinfo.ZoneInfo("Asia/Kolkata")
            except Exception:
                tz = IST
            if getattr(ts.dt, "tz", None) is None:
                ts = ts.dt.tz_localize(tz)
            else:
                ts = ts.dt.tz_convert(tz)
            df["timestamp"] = ts
        return df

    # Public fetch API (daily-only)
    def fetch_daily(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        inst = self._symbol_to_instrument_token(symbol)
        start_dt = dt.datetime.combine(start, dt.time(0, 0, tzinfo=IST))
        end_dt = dt.datetime.combine(end, dt.time(23, 59, tzinfo=IST))
        try:
            rows = self._hist(inst, start_dt, end_dt, interval="day")
        except InvalidInstrument:
            sym = symbol.strip().upper()
            self._instruments.pop(sym, None)
            self._inst_cache_data.pop(sym, None)
            _save_instrument_cache(self._inst_cache_data)
            inst = self._symbol_to_instrument_token(symbol)
            rows = self._hist(inst, start_dt, end_dt, interval="day")
        df = pd.DataFrame(rows)
        if df is None or df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = self._ensure_ist_timestamp(df)
        use_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for c in use_cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df[use_cols]

# -------------------- Helpers: time, indicators, etc. ---------------------
def _ensure_ist(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=False)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST)
    else:
        ts = ts.dt.tz_convert(IST)
    df = df.copy()
    df["timestamp"] = ts
    return df

def _validate_monotonic(df: pd.DataFrame):
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be strictly monotonic increasing")
    if df["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps detected")

def _ensure_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _ema(series: pd.Series, span: int) -> pd.Series:
    return _ensure_float(series).ewm(span=span, adjust=False, min_periods=1).mean()

def _sma(series: pd.Series, window: int) -> pd.Series:
    return _ensure_float(series).rolling(window=window, min_periods=1).mean()

def _rsi(series: pd.Series, period: int) -> pd.Series:
    close = _ensure_float(series)
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    h = _ensure_float(h); l = _ensure_float(l); c = _ensure_float(c)
    pc = c.shift(1)
    ranges = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1)
    return ranges.max(axis=1)

def _atr(h, l, c, period: int) -> pd.Series:
    tr = _true_range(h, l, c)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def _adx(h, l, c, period: int):
    h = _ensure_float(h); l = _ensure_float(l); c = _ensure_float(c)
    up = h.diff()
    dn = l.shift(1) - l
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _true_range(h, l, c)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / denom * 100
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di

def compute_vpoc(df: pd.DataFrame, bins: int = 50) -> float:
    """Simple VPOC: volume-weighted price histogram peak (using typical price)."""
    if df.empty:
        return float("nan")
    vols = df["volume"].to_numpy(dtype="float64")
    if {"high", "low", "close"}.issubset(df.columns):
        highs = df["high"].to_numpy(dtype="float64")
        lows = df["low"].to_numpy(dtype="float64")
        closes = df["close"].to_numpy(dtype="float64")
        prices = (highs + lows + closes) / 3.0
    else:
        prices = df["close"].to_numpy(dtype="float64")
    mask = np.isfinite(prices) & np.isfinite(vols)
    if not mask.any():
        return float("nan")
    prices = prices[mask]; vols = vols[mask]
    lo = float(np.min(prices)); hi = float(np.max(prices))
    if not math.isfinite(lo) or not math.isfinite(hi):
        return float("nan")
    if math.isclose(lo, hi):
        return float(lo)
    hist, edges = np.histogram(prices, bins=bins, range=(lo, hi), weights=vols)
    if hist.size == 0 or np.all(hist == 0):
        return float((lo + hi) / 2.0)
    idx = int(np.argmax(hist))
    up_idx = min(idx + 1, len(edges) - 1)
    return float((edges[idx] + edges[up_idx]) / 2.0)

def _cpr_relationship(base_bc, base_tc, other_bc, other_tc) -> pd.Series:
    relation = pd.Series(index=base_bc.index, dtype="object")
    relation[(other_bc > base_tc) & other_bc.notna() & base_tc.notna()] = "Above"
    relation[(other_tc < base_bc) & other_tc.notna() & base_bc.notna()] = "Below"
    relation[(((other_bc <= base_tc) & (other_tc >= base_bc)) &
              other_bc.notna() & other_tc.notna() & base_bc.notna() & base_tc.notna())] = "Inside"
    relation = relation.fillna("Overlap")
    relation[(base_bc.isna()) | (base_tc.isna()) | (other_bc.isna()) | (other_tc.isna())] = None
    return relation

def _period_trend_from_highs(ts: pd.Series, high: pd.Series, period: str) -> pd.Series:
    if ts.dt.tz is not None:
        ts_local = ts.dt.tz_convert(None)
    else:
        ts_local = ts
    periods = ts_local.dt.to_period(period)
    frame = pd.DataFrame({"period": periods, "high": high})
    agg = frame.groupby("period", sort=True)["high"].max()
    prev_high = agg.shift(1)
    trend = pd.Series(0, index=agg.index, dtype="int8")
    trend[(agg > prev_high) & agg.notna() & prev_high.notna()] = 1
    trend[(agg < prev_high) & agg.notna() & prev_high.notna()] = -1
    return pd.Series(periods.map(trend.to_dict()), index=high.index, dtype="Int8")

DAILY_INDICATOR_COLUMNS = [
    "D_ema20","D_ema50","D_ema100","D_rsi7","D_rsi14",
    "D_macd","D_macd_signal","D_macd_hist","D_cmf20",
    "D_adx14","D_pdi14","D_mdi14","D_inside_day","D_prev_inside_day",
    "D_cpr_pivot","D_cpr_bc","D_cpr_tc","D_pivot","D_support1","D_resistance1",
    "D_support2","D_resistance2","D_nr","D_nr_length","D_nr_day","D_vpoc","D_weekly_vpoc",
    "D_sma5","D_sma20","D_daily_trend","D_weekly_trend","D_monthly_trend",
    "D_rsi7_gt_rsi14","D_ema_stack_20_50_100","D_ema20_angle_deg",
    "D_atr14","D_atr30","D_atr_ratio_14_30","D_cpr_width_pct",
    "D_tmr_cpr_bc","D_tmr_cpr_tc","D_tmr_cpr_vs_today","D_cpr_vs_yday",
    "D_hh","D_hl","D_lh","D_ll","D_structure_trend",
    "D_prev_high","D_prev_low","D_prev_close",
    "D_oli","D_day_type","D_range_to_atr14",
    "D_sma50","D_sma200","D_golden_regime","D_obv","D_obv_slope","D_price_and_obv_rising",
]

def compute_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Pre-create columns (ensures stable schema even for empty frames)
    for col in DAILY_INDICATOR_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series([], dtype="float64")

    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)

    timestamps = pd.to_datetime(df["timestamp"])
    timestamps_local = timestamps.dt.tz_convert(None) if timestamps.dt.tz is not None else timestamps

    close = _ensure_float(df["close"])
    high = _ensure_float(df["high"])
    low = _ensure_float(df["low"])
    open_ = _ensure_float(df["open"])
    volume = _ensure_float(df["volume"]).fillna(0.0)

    # EMAs/RSIs
    df["D_ema20"] = _ema(close, 20); df["D_ema50"] = _ema(close, 50); df["D_ema100"] = _ema(close, 100)
    df["D_rsi7"] = _rsi(close, 7); df["D_rsi14"] = _rsi(close, 14)

    # MACD (12,26,9)
    ema12 = _ema(close, 12); ema26 = _ema(close, 26); macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
    df["D_macd"] = macd; df["D_macd_signal"] = signal; df["D_macd_hist"] = macd - signal

    # CMF (Chaikin Money Flow 20)
    df["D_cmf20"] = (
        (((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume)
        .rolling(window=20, min_periods=1).sum()
        / volume.rolling(window=20, min_periods=1).sum()
    )

    # ADX family (14)
    adx, pdi, mdi = _adx(high, low, close, 14)
    df["D_adx14"] = adx; df["D_pdi14"] = pdi; df["D_mdi14"] = mdi

    # Previous day refs & Inside day flags
    prev_high = high.shift(1); prev_low = low.shift(1); prev_close = close.shift(1)
    df["D_prev_high"] = prev_high; df["D_prev_low"] = prev_low; df["D_prev_close"] = prev_close
    df["D_inside_day"] = ((high <= prev_high) & (low >= prev_low)).astype("boolean")
    df["D_prev_inside_day"] = df["D_inside_day"].shift(1).astype("boolean")

    # CPR / Pivots
    pivot = (high + low + close) / 3
    cpr_bc = (high + low) / 2
    cpr_tc = 2 * pivot - cpr_bc
    df["D_cpr_pivot"] = pivot; df["D_cpr_bc"] = cpr_bc; df["D_cpr_tc"] = cpr_tc

    df["D_pivot"] = pivot
    df["D_support1"] = 2 * pivot - high
    df["D_resistance1"] = 2 * pivot - low
    rng = high - low
    df["D_support2"] = pivot - rng
    df["D_resistance2"] = pivot + rng

    # VPOC (simple) and Weekly VPOC
    df["D_vpoc"] = (high + low + close) / 3.0  # typical price proxy for quick VPOC
    price_profile = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
    weekly_vpoc = pd.Series(np.nan, index=df.index, dtype="float64")
    weekly_periods = timestamps_local.dt.to_period("W-FRI")
    for _, g in price_profile.groupby(weekly_periods):
        weekly_vpoc.loc[g.index] = compute_vpoc(g)
    df["D_weekly_vpoc"] = weekly_vpoc

    # NR series
    prev6_min = rng.shift(1).rolling(window=6, min_periods=6).min()
    nr7 = (rng < prev6_min) & prev6_min.notna()
    df["D_nr"] = nr7.astype("boolean")
    nr_len, count = [], 0
    for flag in df["D_nr"].fillna(False):
        if flag: count += 1
        else: count = 0
        nr_len.append(count)
    df["D_nr_length"] = nr_len

    nr_window = pd.Series(pd.NA, index=df.index, dtype="Int64")
    values = rng.astype("float64")
    for w in range(20, 4, -1):
        prev_min = values.shift(1).rolling(window=w-1, min_periods=w-1).min()
        mask = (values < prev_min) & prev_min.notna()
        nr_window = nr_window.mask(mask & nr_window.isna(), w)
    df["D_nr_day"] = nr_window

    # SMAs / relations / slopes
    df["D_sma5"] = _sma(close, 5); df["D_sma20"] = _sma(close, 20)
    df["D_sma50"] = _sma(close, 50); df["D_sma200"] = _sma(close, 200)

    df["D_rsi7_gt_rsi14"] = (df["D_rsi7"] > df["D_rsi14"]).astype("boolean")
    df["D_ema_stack_20_50_100"] = ((df["D_ema20"] > df["D_ema50"]) & (df["D_ema50"] > df["D_ema100"])).astype("boolean")

    ema20 = df["D_ema20"]; prev_ema20 = ema20.shift(1)
    pct_slope = (ema20 - prev_ema20) / prev_ema20.replace(0, np.nan)
    df["D_ema20_angle_deg"] = np.degrees(np.arctan(pct_slope))

    # ATRs
    df["D_atr14"] = _atr(high, low, close, 14)
    df["D_atr30"] = _atr(high, low, close, 30)
    df["D_atr_ratio_14_30"] = df["D_atr14"] / df["D_atr30"].replace(0, np.nan)

    # CPR widths & relationships
    df["D_cpr_width_pct"] = ((df["D_cpr_tc"] - df["D_cpr_bc"]) / close.replace(0, np.nan)) * 100
    df["D_tmr_cpr_bc"] = cpr_bc.shift(-1)
    df["D_tmr_cpr_tc"] = cpr_tc.shift(-1)
    df["D_tmr_cpr_vs_today"] = _cpr_relationship(cpr_bc, cpr_tc, df["D_tmr_cpr_bc"], df["D_tmr_cpr_tc"])
    df["D_cpr_vs_yday"] = _cpr_relationship(cpr_bc, cpr_tc, cpr_bc.shift(1), cpr_tc.shift(1))

    # Structure/trend flags
    df["D_hh"] = (high > prev_high).astype("boolean")
    df["D_hl"] = (low > prev_low).astype("boolean")
    df["D_lh"] = (high < prev_high).astype("boolean")
    df["D_ll"] = (low < prev_low).astype("boolean")

    daily_trend = pd.Series(0, index=df.index, dtype="int8")
    daily_trend[df["D_hh"] == True] = 1
    daily_trend[df["D_lh"] == True] = -1
    df["D_daily_trend"] = daily_trend.astype("Int8")

    df["D_weekly_trend"] = _period_trend_from_highs(timestamps, high, "W-FRI")
    df["D_monthly_trend"] = _period_trend_from_highs(timestamps, high, "M")

    # Other daily features
    df["D_structure_trend"] = np.select(
        [df["D_hh"] & df["D_hl"], df["D_lh"] & df["D_ll"]],
        ["uptrend", "downtrend"],
        default="range"
    )
    df["D_oli"] = (open_ - low) / rng.replace(0, np.nan)
    df["D_day_type"] = np.select([open_ > cpr_tc, open_ < cpr_bc], ["bullish", "bearish"], default="inside")
    df["D_range_to_atr14"] = rng / df["D_atr14"].replace(0, np.nan)

    df["D_golden_regime"] = ((close > df["D_sma200"]) & (df["D_sma50"] > df["D_sma200"])).astype("boolean")

    # OBV family
    obv = (np.sign(close.diff().fillna(0.0)) * volume).cumsum()
    df["D_obv"] = obv
    df["D_obv_slope"] = df["D_obv"].diff()
    df["D_price_and_obv_rising"] = ((close > close.shift(1)) & (df["D_obv"] > df["D_obv"].shift(1))).astype("boolean")

    return df

# ----------------------------- Daily build --------------------------------
def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()
    df = _ensure_ist(df)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

def _maybe_iso(val):
    if pd.isna(val):
        return None
    if isinstance(val, (pd.Timestamp, dt.datetime)):
        ts = pd.Timestamp(val)
        ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
        return ts.isoformat()
    return str(val)

def _cached_span(path: Path, meta: Optional[dict]) -> Tuple[Optional[dt.date], Optional[dt.date]]:
    meta = meta or {}
    first = _parse_meta_day(meta.get("first_timestamp"))
    last = _parse_meta_day(meta.get("last_timestamp"))
    if (first is None or last is None) and path.exists():
        try:
            ts_df = read_parquet(path, columns=["timestamp"])
        except (ValueError, KeyError):
            ts_df = read_parquet(path)
        if "timestamp" in ts_df.columns and not ts_df.empty:
            ts_df = _ensure_ist(ts_df)
            ts = pd.to_datetime(ts_df["timestamp"], errors="coerce")
            dates = ts.dt.date.dropna()
            if not dates.empty:
                actual_first = dates.min()
                actual_last = dates.max()
            else:
                actual_first = actual_last = None
            if first is None: first = actual_first
            if last is None: last = actual_last
    return first, last

def _parse_meta_day(value) -> Optional[dt.date]:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
        ts = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
        return ts.date()
    except Exception:
        return None

def build_daily(provider: KiteProvider, config: Config, symbol: str, start: dt.date, end: dt.date,
                *, force=False, recompute_only=False) -> Path:
    out_pq = daily_path(config, symbol)
    ok = ok_path(config, symbol)
    ok_meta = read_json(ok)

    # Existing coverage
    cached_first, cached_last = _cached_span(out_pq, ok_meta)
    schema_ok = bool(ok_meta) and ok_meta.get(OK_VERSION_KEY) == SCHEMA_VERSION and out_pq.exists()

    requested_start = _parse_meta_day(ok_meta.get("requested_start")) if ok_meta else None
    requested_end = _parse_meta_day(ok_meta.get("requested_end")) if ok_meta else None
    requested_covers = (
        schema_ok
        and requested_start is not None
        and requested_end is not None
        and requested_start <= start
        and requested_end >= end
    )
    data_covers = (
        schema_ok
        and cached_first is not None
        and cached_last is not None
        and cached_first <= start
        and cached_last >= end
    )

    # Recompute-only: indicators from existing parquet; no refetch
    if recompute_only and out_pq.exists():
        df = _normalize_daily(read_parquet(out_pq))
        if not df.empty:
            _validate_monotonic(df)
            df = compute_daily_indicators(df)
        first_ts = _maybe_iso(df["timestamp"].iloc[0]) if not df.empty else None
        last_ts = _maybe_iso(df["timestamp"].iloc[-1]) if not df.empty else None
        meta = ok_meta_base() | {
            "rows": int(df.shape[0]),
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "requested_start": start.isoformat() if start else None,
            "requested_end": end.isoformat() if end else None,
        }
        with FileLock(out_pq):
            to_parquet(out_pq, df, engine=config.parquet_engine,
                       compression=config.parquet_compression, use_dictionary=config.parquet_use_dictionary)
            write_json_atomic(ok, meta)
        return out_pq

    # --------------------------- INCREMENTAL PATH --------------------------
    # If we have existing cache and not forcing a full rebuild, fetch only tail
    if schema_ok and cached_last is not None and not force:
        fetch_start = cached_last + dt.timedelta(days=1)
        fetch_end = end
        if fetch_start > fetch_end:
            # Nothing new to add
            return out_pq

        base_df = _normalize_daily(read_parquet(out_pq))
        inc_df = _normalize_daily(provider.fetch_daily(symbol, fetch_start, fetch_end))

        if inc_df.empty:
            return out_pq

        merged = pd.concat([base_df, inc_df], ignore_index=True)
        merged = merged.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)

        _validate_monotonic(merged)
        merged = compute_daily_indicators(merged)

        first_ts = _maybe_iso(merged["timestamp"].iloc[0]) if not merged.empty else None
        last_ts = _maybe_iso(merged["timestamp"].iloc[-1]) if not merged.empty else None

        meta = ok_meta_base() | {
            "rows": int(merged.shape[0]),
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "requested_start": start.isoformat() if start else None,
            "requested_end": end.isoformat() if end else None,
        }

        with FileLock(out_pq):
            to_parquet(out_pq, merged, engine=config.parquet_engine,
                       compression=config.parquet_compression, use_dictionary=config.parquet_use_dictionary)
            write_json_atomic(ok, meta)
        return out_pq

    # ----------------------------- FULL FETCH ------------------------------
    # (initial build or when --force is used)
    df = _normalize_daily(provider.fetch_daily(symbol, start, end))
    if not df.empty:
        _validate_monotonic(df)
        df = compute_daily_indicators(df)

    first_ts = _maybe_iso(df["timestamp"].iloc[0]) if not df.empty else None
    last_ts = _maybe_iso(df["timestamp"].iloc[-1]) if not df.empty else None

    with FileLock(out_pq):
        to_parquet(out_pq, df, engine=config.parquet_engine,
                   compression=config.parquet_compression, use_dictionary=config.parquet_use_dictionary)
        write_json_atomic(ok, ok_meta_base() | {
            "rows": int(df.shape[0]),
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "requested_start": start.isoformat() if start else None,
            "requested_end": end.isoformat() if end else None,
        })
    return out_pq

# ----------------------------- Orchestrator -------------------------------
def parse_date_input(value: Optional[str | dt.date]) -> dt.date:
    if isinstance(value, dt.date):
        return value
    if value is None:
        raise ValueError("Date value is required")
    text = str(value).strip()
    if not text:
        raise ValueError("Date value is required")
    formats = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y")
    for fmt in formats:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: {text!r}")

def normalize_requested_range(start_value: Optional[str | dt.date], end_value: Optional[str | dt.date]) -> Tuple[dt.date, dt.date, List[str]]:
    start = parse_date_input(start_value)
    end = parse_date_input(end_value)
    if start > end:
        start, end = end, start
    notes: List[str] = []
    today = today_ist()
    if end > today:
        notes.append(
            f"End date {end.isoformat()} trimmed to {today.isoformat()} because future data is unavailable."
        )
        end = today
    if start > today:
        notes.append(
            f"Start date {start.isoformat()} adjusted to {today.isoformat()} because the market has not traded yet."
        )
        start = today
    if start > end:
        raise ValueError("Requested date range does not contain any trading days after adjustments.")
    return start, end, notes

def append_token_error_log(base_dir: Path, *, symbol: str, phase: str, error: str) -> None:
    log_path = (base_dir / "_token_expired.log") if base_dir else Path("_token_expired.log")
    rec = {
        "ts": dt.datetime.now(tz=IST).isoformat(),
        "symbol": symbol,
        "day": None,
        "phase": phase,
        "error": str(error),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

class Pipeline:
    def __init__(self, provider: KiteProvider, config: Config, progress_cb: Optional[callable] = None):
        self.provider = provider
        self.cfg = config
        self.ratelimiter = RateLimiter(config.rate_limit_per_sec)
        self._daily_cache = DataFrameCache(maxsize=256)
        self.fetch_daily = with_retry(self._cached_fetch_daily, tries=config.retry_tries, backoff=config.retry_backoff_base)
        self.progress_cb = progress_cb or (lambda msg: None)

    def _cached_fetch_daily(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        key = (symbol, start, end)
        cached = self._daily_cache.get(key)
        if cached is not None:
            return cached
        self.ratelimiter.acquire()
        df = self.provider.fetch_daily(symbol, start, end)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("fetch_daily must return a DataFrame")
        return self._daily_cache.put(key, df)

    def build(self, symbols: Sequence[str], start_date: dt.date, end_date: dt.date, *, force: bool, recompute_only: bool):
        cfg = self.cfg
        start_date, end_date, adjustments = normalize_requested_range(start_date, end_date)
        for note in adjustments:
            try:
                self.progress_cb(f"NOTE: {note}")
            except Exception:
                print(note)
        # Pre-resolve instruments (helpful for early failures)
        pre = []
        unresolved: List[str] = []
        for s in symbols:
            try:
                _ = self.provider._symbol_to_instrument_token(s)
                pre.append(s)
            except UnresolvedSymbol as e:
                unresolved.append(str(e))
                append_token_error_log(cfg.day_root(), symbol=str(e), phase="preresolve", error="UNRESOLVED_SYMBOL")
                continue
        symbols = pre

        # Daily build only
        daily_start = start_date - dt.timedelta(days=5)  # a small buffer is OK; incremental logic will clamp
        daily_end = end_date
        for s in symbols:
            try:
                build_daily(self.provider, cfg, s, daily_start, daily_end, force=force, recompute_only=recompute_only)
            except UnresolvedSymbol:
                self.progress_cb(f"SKIP unresolved: {s} (daily)")
                continue
            except AuthExpired as e:
                append_token_error_log(cfg.day_root(), symbol=s, phase="daily", error=str(e))
                raise
            self.progress_cb(f"Daily: {s}")

        if unresolved:
            summary = f"Unresolved symbols skipped ({len(unresolved)}): " + ", ".join(sorted(set(unresolved)))
            print(summary)
            try:
                if TK_OK:
                    tk.messagebox.showwarning("Unresolved", summary)
            except Exception:
                pass

# -------------------------- Symbols file loader ---------------------------
def _load_symbols_from_file(path: str) -> List[str]:
    p = Path(path)
    ext = p.suffix.lower()
    items: List[str] = []
    if ext in (".xls", ".xlsx"):
        df = pd.read_excel(p)  # requires openpyxl installed
        if df.empty:
            return []
        col0 = df.columns[0]
        items = [str(x) for x in df[col0].dropna().tolist()]
    else:
        try:
            df = pd.read_csv(p, header=None)
            items = [str(x) for x in df.iloc[:, 0].dropna().tolist()] if df.shape[1] >= 1 else []
        except Exception:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            tokens = [t.strip() for t in raw.replace("\n", ",").replace("\t", ",").split(",")]
            items = [t for t in tokens if t]
    cleaned, seen = [], set()
    for t in items:
        s = sanitize_symbol(t)
        if s and s.casefold() not in seen:
            seen.add(s.casefold()); cleaned.append(s)
    return cleaned

# --------------------------- GUI inputs (FILE) ----------------------------
def _ask_user_inputs_gui_file_only():
    if not TK_OK:
        raise SystemExit("Tkinter is not available.")
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)

    # File with symbols
    path = filedialog.askopenfilename(
        title="Select symbols file (Excel/CSV/TXT; first column = symbols)",
        filetypes=[("Excel/CSV/TXT", "*.xlsx *.xls *.csv *.txt"), ("All", "*.*")]
    )
    if not path:
        messagebox.showerror("Required", "No symbols file selected.")
        raise SystemExit(1)

    symbols = _load_symbols_from_file(path)
    if not symbols:
        messagebox.showerror("Invalid file", "Could not parse any symbols from the file.")
        raise SystemExit(1)

    today = today_ist()
    default_start = today - dt.timedelta(days=600)
    start_input = simpledialog.askstring(
        "Start date",
        "Enter start date (YYYY-MM-DD or DD-MM-YYYY):",
        initialvalue=default_start.isoformat(),
    )
    end_input = simpledialog.askstring(
        "End date",
        "Enter end date (YYYY-MM-DD or DD-MM-YYYY):",
        initialvalue=today.isoformat(),
    )

    try:
        start_date, end_date, adjustments = normalize_requested_range(start_input, end_input)
    except Exception as exc:
        messagebox.showerror("Invalid dates", str(exc))
        raise SystemExit(1)

    if adjustments:
        messagebox.showinfo("Adjusted dates", "\n".join(adjustments))

    mw = simpledialog.askinteger(
        "Parallel workers",
        "Max threads (IO-bound; 12–32 works well):",
        initialvalue=24, minvalue=1, maxvalue=128
    )
    if not mw:
        mw = 24

    base_config = Config.from_env()
    messagebox.showinfo(
        "Summary",
        (
            "Symbols file: {}\n\nSymbols parsed: {}\n\n"
            "Date range: {} → {}\nWorkers: {}\n\n"
            "Daily folder:\n{}"
        ).format(
            path, len(symbols),
            start_date.isoformat(), end_date.isoformat(), mw,
            str(base_config.daily_root),
        ),
    )
    root.destroy()

    return {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "max_workers": mw,
        "base_config": base_config,
    }

# --------------------------------- CLI ------------------------------------
def parse_cli_args():
    import argparse
    ap = argparse.ArgumentParser(description="FAST daily-only cache builder for Kite (no intraday).")
    ap.add_argument("--symbols", nargs="*", help="Symbols list (space separated).", default=None)
    ap.add_argument("--symbols-file", help="Path to Excel/CSV/TXT with first column = symbols.", default=None)
    ap.add_argument("--start-date", help="Start date (YYYY-MM-DD).", default=None)
    ap.add_argument("--end-date", help="End date (YYYY-MM-DD).", default=None)
    ap.add_argument("--days", type=int, default=None, help="Fallback: number of recent calendar days (converted to trading days).")
    ap.add_argument("--workers", type=int, default=24, help="Thread pool size (default 24).")
    ap.add_argument("--rate", type=float, default=12.0, help="Requests per second (default 12).")
    ap.add_argument("--force", action="store_true", help="Force rebuild ignoring .ok coverage.")
    ap.add_argument("--recompute-only", action="store_true", help="Recompute indicators from existing daily parquet without refetching.")
    ap.add_argument("--incremental", action="store_true", help="Incrementally update caches from last cached date to today.")
    args = ap.parse_args()
    syms: Optional[List[str]] = args.symbols
    if args.symbols_file:
        syms = _load_symbols_from_file(args.symbols_file)
    return args, syms

def _resolve_requested_dates(start_value: Optional[str], end_value: Optional[str], days_value: Optional[int],
                             *, incremental: bool = False) -> Tuple[dt.date, dt.date]:
    if incremental:
        # Provide a generous window (last ~600 days → today); build_daily will reduce the fetch
        # to only cached_last+1 → end for each symbol.
        today = today_ist()
        start = today - dt.timedelta(days=600)
        start, end, _ = normalize_requested_range(start, today)
        return start, end
    if start_value and end_value:
        start, end, _ = normalize_requested_range(start_value, end_value)
        return start, end
    if days_value and days_value > 0:
        # Approx recent range → then clamp to today if needed
        today = today_ist()
        start = today - dt.timedelta(days=days_value)
        start, end, _ = normalize_requested_range(start, today)
        return start, end
    raise SystemExit("Provide --start-date/--end-date or --days to define the caching window.")

# -------------------------------- Entry -----------------------------------
def main():
    try:
        if len(sys.argv) > 1:
            args, symbols = parse_cli_args()
            if not symbols:
                raise SystemExit("No symbols provided. Use --symbols-file <path> or --symbols ...")
            base_config = Config.from_env()
            cfg = base_config.with_updates(
                max_workers=int(args.workers),
                rate_limit_per_sec=float(args.rate),
                request_timeout_s=15.0,
                retry_tries=6,
            )
            provider = KiteProvider()
            pipeline = Pipeline(provider, cfg, progress_cb=lambda m: print(m))
            cfg.daily_root.mkdir(parents=True, exist_ok=True)
            start_date, end_date = _resolve_requested_dates(args.start_date, args.end_date, args.days,
                                                            incremental=bool(args.incremental))
            pipeline.build(symbols, start_date, end_date, force=bool(args.force), recompute_only=bool(args.recompute_only))
            print("FAST daily-only cache build completed.")
        else:
            # GUI path (file-only symbols)
            ui = _ask_user_inputs_gui_file_only()
            base_config = ui.get("base_config") or Config.from_env()
            cfg = base_config.with_updates(
                max_workers=int(ui["max_workers"]),
                rate_limit_per_sec=12.0,
                request_timeout_s=15.0,
                retry_tries=6,
            )
            symbols = ui["symbols"]
            start_date = ui["start_date"]
            end_date = ui["end_date"]

            if not TK_OK:
                raise SystemExit("Tkinter not available; GUI mode is required.")

            root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)

            class ProgressUI:
                def __init__(self, total: int):
                    self.total = max(1, int(total))
                    self.start = time.perf_counter()
                    self.completed = 0
                    self.root = tk.Toplevel()
                    self.root.title("Building daily cache (FAST)...")
                    self.root.geometry("560x160")
                    self.root.resizable(False, False)
                    self.label = tk.Label(self.root, text="Starting...", anchor="w")
                    self.label.pack(fill="x", padx=12, pady=(12, 6))
                    self.pb = ttk.Progressbar(self.root, orient="horizontal", mode="determinate", maximum=self.total, length=520)
                    self.pb.pack(padx=12, pady=6)
                    self.eta = tk.Label(self.root, text="ETA: --:--", anchor="w")
                    self.eta.pack(fill="x", padx=12, pady=(6, 12))
                    self.root.attributes("-topmost", True)
                    self.root.update_idletasks()

                def _fmt_eta(self, secs: float) -> str:
                    if secs is None or secs != secs or secs == float("inf"):
                        return "--:--"
                    m, s = divmod(int(secs), 60); h, m = divmod(m, 60)
                    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

                def tick(self, msg: str = ""):
                    self.completed += 1
                    self.pb["value"] = self.completed
                    elapsed = max(0.001, time.perf_counter() - self.start)
                    rate = self.completed / elapsed
                    remaining = max(0, self.total - self.completed)
                    eta_s = remaining / rate if rate > 0 else None
                    self.label.config(text=msg or f"Completed {self.completed}/{self.total}")
                    self.eta.config(text=f"ETA: {self._fmt_eta(eta_s)} Elapsed: {self._fmt_eta(elapsed)}")
                    self.root.update_idletasks()

                def done(self):
                    self.pb["value"] = self.total
                    self.label.config(text=f"Done: {self.total}/{self.total}")
                    self.eta.config(text=f"ETA: 00:00 Elapsed: {self._fmt_eta(time.perf_counter() - self.start)}")
                    self.root.update_idletasks()
            pui = ProgressUI(total=len(symbols))
            def on_progress(msg: str): pui.tick(msg)

            provider = KiteProvider()
            pipeline = Pipeline(provider, cfg, progress_cb=on_progress)
            cfg.daily_root.mkdir(parents=True, exist_ok=True)

            # GUI path: provide a broad range; incremental logic within build_daily
            pipeline.build(symbols, start_date, end_date, force=False, recompute_only=False)
            pui.done()
            messagebox.showinfo("Done", "FAST daily-only cache build completed.")
    except Exception as e:
        if TK_OK:
            messagebox.showerror("Error", str(e))
        else:
            print("ERROR:", e, file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
