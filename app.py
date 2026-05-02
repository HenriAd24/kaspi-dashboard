"""
Stock Dashboard — Live price + historical P/E corridor.
Supports any NASDAQ/NYSE ticker. Auto-detects currency conversion.
"""

from flask import Flask, jsonify, render_template, send_from_directory, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import time
import logging
import sys
import socket
import re
import os

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

app = Flask(__name__)

DEFAULT_TICKER = "KSPI"
TICKER_RE = re.compile(r'^[A-Z0-9.\-\^]{1,10}$')

_cache: dict = {}
_lock = threading.Lock()


def cache_get(key: str, ttl: int):
    with _lock:
        e = _cache.get(key)
        if e and (time.time() - e["ts"]) < ttl:
            return e["data"]
    return None


def cache_set(key: str, data):
    with _lock:
        _cache[key] = {"data": data, "ts": time.time()}


def _clean_ticker(raw: str) -> str:
    t = re.sub(r'\s+', '', raw).upper()[:10]
    if not TICKER_RE.match(t):
        raise ValueError(f"Invalid ticker: {raw!r}")
    return t


def _get_ticker() -> str:
    try:
        return _clean_ticker(request.args.get("ticker", DEFAULT_TICKER))
    except ValueError:
        return DEFAULT_TICKER


def _tz_strip(idx):
    try:
        return idx.tz_localize(None)
    except TypeError:
        return idx.tz_convert(None)


def _safe_info(t: yf.Ticker, ticker: str) -> dict:
    try:
        raw = t.info
        if raw and len(raw) > 5:
            log.info("[%s] t.info OK (%d fields)", ticker, len(raw))
            return raw
    except Exception as e:
        log.warning("[%s] t.info failed: %s", ticker, e)
    return {}


def _safe_price(t: yf.Ticker, info: dict, ticker: str) -> float | None:
    p = info.get("regularMarketPrice") or info.get("currentPrice")
    if p:
        return float(p)
    try:
        p = float(t.fast_info.last_price)
        log.info("[%s] price from fast_info: %.2f", ticker, p)
        return p
    except Exception as e:
        log.warning("[%s] fast_info.last_price: %s", ticker, e)
    return None


# ---------------------------------------------------------------------------
# Live price
# ---------------------------------------------------------------------------

def fetch_live_price(ticker: str) -> dict | None:
    t = yf.Ticker(ticker)
    try:
        fi = t.fast_info
        price = float(fi.last_price)
        prev  = float(fi.previous_close)
        return {
            "price":      round(price, 2),
            "change":     round(price - prev, 2),
            "change_pct": round((price - prev) / prev * 100, 2),
            "prev_close": round(prev, 2),
            "day_high":   round(float(fi.day_high), 2) if fi.day_high else None,
            "day_low":    round(float(fi.day_low),  2) if fi.day_low  else None,
            "volume":     int(fi.last_volume)           if fi.last_volume else None,
            "market_cap": int(fi.market_cap)            if fi.market_cap  else None,
            "currency":   getattr(fi, "currency", "USD"),
            "timestamp":  datetime.now().isoformat(),
        }
    except Exception as e:
        log.warning("[%s] fast_info live: %s", ticker, e)

    try:
        info  = _safe_info(t, ticker)
        price = float(info["regularMarketPrice"])
        prev  = float(info["previousClose"])
        return {
            "price":      round(price, 2),
            "change":     round(price - prev, 2),
            "change_pct": round((price - prev) / prev * 100, 2),
            "prev_close": round(prev, 2),
            "day_high":   info.get("dayHigh"),
            "day_low":    info.get("dayLow"),
            "volume":     info.get("regularMarketVolume"),
            "market_cap": info.get("marketCap"),
            "currency":   info.get("currency", "USD"),
            "timestamp":  datetime.now().isoformat(),
        }
    except Exception as e:
        log.error("[%s] live price failed: %s", ticker, e)
    return None


# ---------------------------------------------------------------------------
# Company info
# ---------------------------------------------------------------------------

def fetch_company_info(ticker: str) -> dict:
    t    = yf.Ticker(ticker)
    info = _safe_info(t, ticker)

    fi_data = {}
    try:
        fi = t.fast_info
        fi_data = {
            "market_cap": int(fi.market_cap)                     if fi.market_cap                else None,
            "year_high":  round(float(fi.year_high), 2)          if fi.year_high                 else None,
            "year_low":   round(float(fi.year_low),  2)          if fi.year_low                  else None,
            "avg_volume": int(fi.three_month_average_volume)     if fi.three_month_average_volume else None,
            "shares":     int(fi.shares)                         if fi.shares                    else None,
            "price":      round(float(fi.last_price), 2)         if fi.last_price                else None,
        }
    except Exception as e:
        log.warning("[%s] fast_info company_info: %s", ticker, e)

    trailing_eps = info.get("trailingEps")
    trailing_pe  = info.get("trailingPE")

    # Best fallback: read from PE cache (computed first in warmup)
    if not trailing_pe or not trailing_eps:
        pe_cached = cache_get(f"{ticker}:pe", 7200)
        if pe_cached:
            if not trailing_pe:
                trailing_pe = pe_cached.get("current_pe")
            if not trailing_eps and trailing_pe and fi_data.get("price"):
                trailing_eps = round(fi_data["price"] / trailing_pe, 2)
            if trailing_pe:
                log.info("[%s] EPS/PE from PE cache: pe=%.2f", ticker, trailing_pe)

    # Last resort: earnings_history
    if not trailing_eps:
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty and "epsActual" in eh.columns:
                vals = [v for v in eh["epsActual"].tail(4).tolist() if not pd.isna(v)]
                if vals:
                    ttm   = sum(float(v) for v in vals)
                    price = fi_data.get("price", 0)
                    if price > 0 and ttm != 0:
                        impl_pe = price / ttm
                        if 2 < impl_pe < 200:
                            trailing_eps = round(ttm, 2)  # already USD
                        else:
                            try:
                                rate = float(yf.Ticker("USDKZT=X").fast_info.last_price)
                                trailing_eps = round(ttm / rate, 2)
                            except Exception:
                                pass
        except Exception as e:
            log.warning("[%s] EPS earnings fallback: %s", ticker, e)

    if not trailing_pe and trailing_eps and fi_data.get("price"):
        trailing_pe = round(fi_data["price"] / trailing_eps, 2)

    mc = info.get("marketCap") or fi_data.get("market_cap")
    return {
        "name":           info.get("longName") or info.get("shortName") or ticker,
        "exchange":       info.get("exchange", ""),
        "sector":         info.get("sector"),
        "industry":       info.get("industry"),
        "market_cap":     mc,
        "market_cap_fmt": f"${mc/1e9:.1f}B" if mc else "-",
        "trailing_pe":    trailing_pe,
        "forward_pe":     info.get("forwardPE"),
        "trailing_eps":   trailing_eps,
        "peg_ratio":      info.get("pegRatio"),
        "dividend_yield": info.get("dividendYield"),
        "fifty2_high":    info.get("fiftyTwoWeekHigh") or fi_data.get("year_high"),
        "fifty2_low":     info.get("fiftyTwoWeekLow")  or fi_data.get("year_low"),
        "avg_volume":     info.get("averageVolume")     or fi_data.get("avg_volume"),
        "beta":           info.get("beta"),
        "shares_out":     info.get("sharesOutstanding") or fi_data.get("shares"),
        "timestamp":      datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Historical P/E
# ---------------------------------------------------------------------------

def _build_eps_steps(t: yf.Ticker, info: dict, ticker: str) -> list:
    trailing_eps_usd = info.get("trailingEps")

    shares = (info.get("sharesOutstanding") or
              info.get("impliedSharesOutstanding") or 0)
    if not shares:
        try:
            shares = int(t.fast_info.shares)
        except Exception:
            pass

    # --- Collect quarterly EPS (may be in local currency) ---
    quarterly: dict[str, float] = {}

    try:
        eh = t.earnings_history
        if eh is not None and not eh.empty and "epsActual" in eh.columns:
            for dt, row in eh.iterrows():
                v = row.get("epsActual")
                if v is not None and not pd.isna(v):
                    quarterly[str(dt)[:10]] = float(v)
            log.info("[%s] %d quarters from earnings_history", ticker, len(quarterly))
    except Exception as e:
        log.debug("[%s] earnings_history: %s", ticker, e)

    if not quarterly:
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                col = next((c for c in ed.columns if "reported" in c.lower()), None)
                if col:
                    for dt, row in ed.iterrows():
                        v = row.get(col)
                        if v is not None and not pd.isna(v) and float(v) > 0:
                            quarterly[str(dt)[:10]] = float(v)
                    log.info("[%s] %d quarters from earnings_dates", ticker, len(quarterly))
        except Exception as e:
            log.debug("[%s] earnings_dates: %s", ticker, e)

    if not quarterly and shares:
        try:
            qf = t.quarterly_financials
            if qf is not None and "Net Income" in qf.index:
                for dt, ni in qf.loc["Net Income"].dropna().items():
                    quarterly[str(dt)[:10]] = float(ni) / shares
                log.info("[%s] %d quarters from quarterly_financials", ticker, len(quarterly))
        except Exception as e:
            log.debug("[%s] quarterly_financials: %s", ticker, e)

    if not quarterly:
        log.warning("[%s] No quarterly EPS data", ticker)
        return []

    # --- Auto-detect currency conversion factor ---
    sorted_q   = sorted(quarterly.items())
    eps_factor = 1.0  # multiply local EPS → USD

    if trailing_eps_usd and len(sorted_q) >= 4:
        last4 = sum(v for _, v in sorted_q[-4:])
        if last4 > 0 and trailing_eps_usd > 0:
            ratio = trailing_eps_usd / last4
            if 0.5 < ratio < 2.0:
                eps_factor = 1.0
                log.info("[%s] EPS in USD (ratio=%.3f)", ticker, ratio)
            else:
                eps_factor = ratio
                log.info("[%s] Currency factor=%.6f (trailing=%.2f, last4=%.2f)",
                         ticker, ratio, trailing_eps_usd, last4)

    # --- Build TTM steps ---
    steps: list[tuple[str, float]] = []
    dates = [d for d, _ in sorted_q]
    vals  = [v for _, v in sorted_q]

    for i in range(3, len(sorted_q)):
        ttm_usd = sum(vals[i-3:i+1]) * eps_factor
        if ttm_usd > 0:
            steps.append((dates[i], round(ttm_usd, 4)))

    log.info("[%s] Built %d TTM steps", ticker, len(steps))

    # --- Annual anchors for early coverage ---
    first_step = steps[0][0] if steps else "9999-12-31"
    try:
        ist = t.income_stmt
        if ist is not None and "Net Income" in ist.index and shares:
            for dt, ni in ist.loc["Net Income"].dropna().items():
                yr_str  = str(dt)[:10]
                ann_usd = (float(ni) / shares) * eps_factor
                if yr_str < first_step and ann_usd > 0:
                    steps.append((yr_str, round(ann_usd, 4)))
                    log.info("[%s] Annual anchor %s: EPS=$%.2f", ticker, yr_str, ann_usd)
    except Exception as e:
        log.debug("[%s] income_stmt anchors: %s", ticker, e)

    steps.sort(key=lambda x: x[0])
    return steps


def fetch_historical_pe(ticker: str) -> dict | None:
    t     = yf.Ticker(ticker)
    info  = _safe_info(t, ticker)

    current_price = _safe_price(t, info, ticker)
    if not current_price:
        return None

    current_pe  = info.get("trailingPE")
    current_eps = info.get("trailingEps")

    eps_steps = _build_eps_steps(t, info, ticker)

    if not eps_steps:
        if current_pe and current_pe > 0:
            return {
                "pe_history": [], "current_pe": round(float(current_pe), 2),
                "val_label": "–", "val_color": "#94a3b8", "stats": {},
                "eps_source": "current_only", "timestamp": datetime.now().isoformat(),
            }
        return None

    if not current_pe and eps_steps and current_price:
        latest_eps = eps_steps[-1][1]
        if latest_eps > 0:
            current_pe  = round(current_price / latest_eps, 2)
            current_eps = latest_eps

    hist = yf.Ticker(ticker).history(period="5y", interval="1d")
    hist.index = _tz_strip(hist.index)
    hist = hist.sort_index()
    if hist.empty:
        return None

    pe_rows = []
    step_idx = 0
    for date, row in hist.iterrows():
        date_str = str(date)[:10]
        while (step_idx + 1 < len(eps_steps) and
               eps_steps[step_idx + 1][0] <= date_str):
            step_idx += 1
        if eps_steps[step_idx][0] > date_str:
            continue
        ttm_eps = eps_steps[step_idx][1]
        if ttm_eps <= 0:
            continue
        price = float(row["Close"])
        pe    = price / ttm_eps
        if 0 < pe < 500:
            pe_rows.append({
                "date":    date_str,
                "pe":      round(pe, 2),
                "price":   round(price, 2),
                "ttm_eps": round(ttm_eps, 4),
            })

    if current_pe and current_pe > 0 and current_price:
        today = datetime.now().strftime("%Y-%m-%d")
        if not pe_rows or pe_rows[-1]["date"] < today:
            pe_rows.append({
                "date":    today,
                "pe":      round(float(current_pe), 2),
                "price":   round(float(current_price), 2),
                "ttm_eps": round(float(current_eps or current_price / current_pe), 4),
            })

    pe_rows.sort(key=lambda x: x["date"])
    if not pe_rows:
        return None

    vals = [r["pe"] for r in pe_rows]
    pcts = np.percentile(vals, [10, 25, 50, 75, 90])

    val_label, val_color = "–", "#64748b"
    if current_pe:
        cp = float(current_pe)
        if   cp <= pcts[1]: val_label, val_color = "Günstig",         "#16a34a"
        elif cp <= pcts[2]: val_label, val_color = "Leicht günstig",  "#65a30d"
        elif cp <= pcts[3]: val_label, val_color = "Fair bewertet",   "#ca8a04"
        elif cp <= pcts[4]: val_label, val_color = "Leicht teuer",    "#ea580c"
        else:               val_label, val_color = "Teuer",           "#dc2626"

    return {
        "pe_history":  pe_rows,
        "current_pe":  round(float(current_pe), 2) if current_pe else None,
        "val_label":   val_label,
        "val_color":   val_color,
        "stats": {
            "count":  len(vals),
            "min":    round(float(np.min(vals)),  2),
            "p10":    round(float(pcts[0]),        2),
            "p25":    round(float(pcts[1]),        2),
            "median": round(float(pcts[2]),        2),
            "p75":    round(float(pcts[3]),        2),
            "p90":    round(float(pcts[4]),        2),
            "max":    round(float(np.max(vals)),  2),
            "mean":   round(float(np.mean(vals)), 2),
        },
        "eps_source":  "quarterly_step_function",
        "timestamp":   datetime.now().isoformat(),
    }


def fetch_price_history(ticker: str) -> list:
    try:
        hist = yf.Ticker(ticker).history(period="2y", interval="1d")
        hist.index = _tz_strip(hist.index)
        rows = []
        for date, row in hist.iterrows():
            rows.append({
                "date":   str(date)[:10],
                "open":   round(float(row["Open"]),  2),
                "high":   round(float(row["High"]),  2),
                "low":    round(float(row["Low"]),   2),
                "close":  round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
        return rows
    except Exception as e:
        log.error("[%s] price history: %s", ticker, e)
        return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", default_ticker=DEFAULT_TICKER)


@app.route("/sw.js")
def service_worker():
    resp = send_from_directory("static", "sw.js")
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json")


@app.route("/api/live")
def api_live():
    ticker = _get_ticker()
    key    = f"{ticker}:live"
    data   = cache_get(key, 30)
    if data is None:
        data = fetch_live_price(ticker)
        if data:
            cache_set(key, data)
    if data:
        return jsonify(data)
    return jsonify({"error": "Daten nicht verfügbar"}), 503


@app.route("/api/info")
def api_info():
    ticker = _get_ticker()
    key    = f"{ticker}:info"
    data   = cache_get(key, 3600)
    if data is None:
        data = fetch_company_info(ticker)
        if data:
            cache_set(key, data)
    return jsonify(data or {"error": "Info nicht verfügbar"})


@app.route("/api/pe")
def api_pe():
    ticker = _get_ticker()
    key    = f"{ticker}:pe"
    data   = cache_get(key, 3600)
    if data is None:
        log.info("[%s] Computing historical P/E ...", ticker)
        data = fetch_historical_pe(ticker)
        if data:
            cache_set(key, data)
    if data:
        return jsonify(data)
    return jsonify({"error": "KGV-Daten nicht verfügbar"}), 503


@app.route("/api/price-history")
def api_price_history():
    ticker = _get_ticker()
    key    = f"{ticker}:ph"
    data   = cache_get(key, 3600)
    if data is None:
        data = fetch_price_history(ticker)
        if data:
            cache_set(key, data)
    return jsonify(data or [])


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "cached": list(_cache.keys()),
                    "time": datetime.now().isoformat()})


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def _warmup():
    time.sleep(2)
    t = DEFAULT_TICKER
    log.info("Warmup: %s", t)
    d = fetch_live_price(t);      d and cache_set(f"{t}:live", d)
    d = fetch_historical_pe(t);   d and cache_set(f"{t}:pe", d)
    d = fetch_company_info(t);    d and cache_set(f"{t}:info", d)
    log.info("Warmup done: %s", t)


threading.Thread(target=_warmup, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*48}\n  Stock Dashboard\n{'='*48}")
    print(f"  http://localhost:{port}")
    print(f"  http://{local_ip}:{port}  (Handy, gleiches WLAN)")
    print(f"{'='*48}\n")
    app.run(debug=False, port=port, host="0.0.0.0")
