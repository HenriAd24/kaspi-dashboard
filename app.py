"""
Kaspi.kz (KSPI) Stock Dashboard
================================
Live price + historical P/E corridor chart.

KSPI-specific notes:
  - Listed on NASDAQ since January 2024.
  - yfinance reports financials in KZT but price/trailingEps in USD.
  - We derive a KZT->USD EPS conversion factor from trailingEps (USD) /
    sum(last 4 quarters actual KZT EPS).
  - A step-function TTM EPS timeline is built from all available quarterly
    and annual data, then applied to the full daily price history (~580 rows)
    to produce a rich daily P/E series for the corridor chart.

Fallback chain (live price):
  fast_info -> info dict -> latest history candle
"""

from flask import Flask, jsonify, render_template, send_from_directory, Response
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import time
import logging
import sys
import socket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

app = Flask(__name__)

TICKER       = "KSPI"
COMPANY_NAME = "Kaspi.kz"

# Thread-safe in-memory cache
_cache: dict = {}
_lock  = threading.Lock()


def cache_get(key: str, ttl: int):
    with _lock:
        e = _cache.get(key)
        if e and (time.time() - e["ts"]) < ttl:
            return e["data"]
    return None


def cache_set(key: str, data):
    with _lock:
        _cache[key] = {"data": data, "ts": time.time()}


def _tz_strip(idx):
    """Return timezone-naive DatetimeIndex regardless of input tz."""
    try:
        return idx.tz_localize(None)
    except TypeError:
        return idx.tz_convert(None)


def _ticker() -> yf.Ticker:
    return yf.Ticker(TICKER)


# ---------------------------------------------------------------------------
# Live price
# ---------------------------------------------------------------------------

def fetch_live_price() -> dict | None:
    """Three-level fallback for current price."""

    # Method 1: fast_info
    try:
        t  = _ticker()
        fi = t.fast_info
        price = float(fi.last_price)
        prev  = float(fi.previous_close)
        return {
            "price":       round(price, 2),
            "change":      round(price - prev, 2),
            "change_pct":  round((price - prev) / prev * 100, 2),
            "prev_close":  round(prev, 2),
            "day_high":    round(float(fi.day_high), 2) if fi.day_high else None,
            "day_low":     round(float(fi.day_low),  2) if fi.day_low  else None,
            "volume":      int(fi.last_volume) if fi.last_volume else None,
            "market_cap":  int(fi.market_cap)  if fi.market_cap  else None,
            "currency":    "USD",
            "timestamp":   datetime.now().isoformat(),
            "source":      "fast_info",
        }
    except Exception as e:
        log.warning("fast_info failed: %s", e)

    # Method 2: full info dict
    try:
        info  = _ticker().info
        price = float(info["regularMarketPrice"])
        prev  = float(info["previousClose"])
        return {
            "price":       round(price, 2),
            "change":      round(price - prev, 2),
            "change_pct":  round((price - prev) / prev * 100, 2),
            "prev_close":  round(prev, 2),
            "day_high":    info.get("dayHigh"),
            "day_low":     info.get("dayLow"),
            "volume":      info.get("regularMarketVolume"),
            "market_cap":  info.get("marketCap"),
            "currency":    info.get("currency", "USD"),
            "timestamp":   datetime.now().isoformat(),
            "source":      "info",
        }
    except Exception as e:
        log.warning("info fetch failed: %s", e)

    # Method 3: last history candle
    try:
        hist  = _ticker().history(period="5d", interval="1d")
        price = float(hist["Close"].iloc[-1])
        prev  = float(hist["Close"].iloc[-2])
        return {
            "price":      round(price, 2),
            "change":     round(price - prev, 2),
            "change_pct": round((price - prev) / prev * 100, 2),
            "prev_close": round(prev, 2),
            "currency":   "USD",
            "timestamp":  datetime.now().isoformat(),
            "source":     "history_fallback",
        }
    except Exception as e:
        log.error("All price methods failed: %s", e)

    return None


# ---------------------------------------------------------------------------
# Company info
# ---------------------------------------------------------------------------

def fetch_company_info() -> dict:
    t    = _ticker()
    info = {}
    try:
        raw = t.info
        if raw and len(raw) > 5:
            info = raw
            log.info("t.info OK (%d fields)", len(info))
    except Exception as e:
        log.warning("t.info failed in fetch_company_info: %s", e)

    # Always try fast_info — it works even when t.info is blocked
    fi_data = {}
    try:
        fi = t.fast_info
        fi_data = {
            "market_cap": int(fi.market_cap)                    if fi.market_cap               else None,
            "year_high":  round(float(fi.year_high), 2)         if fi.year_high                else None,
            "year_low":   round(float(fi.year_low),  2)         if fi.year_low                 else None,
            "avg_volume": int(fi.three_month_average_volume)    if fi.three_month_average_volume else None,
            "shares":     int(fi.shares)                        if fi.shares                   else None,
            "price":      round(float(fi.last_price), 2)        if fi.last_price               else None,
        }
        log.info("fast_info OK: price=$%.2f mc=$%.1fB",
                 fi_data.get("price", 0), (fi_data.get("market_cap") or 0) / 1e9)
    except Exception as e:
        log.warning("fast_info failed in fetch_company_info: %s", e)

    # Trailing EPS / PE — try info first, then PE cache, then earnings_history
    trailing_eps = info.get("trailingEps")
    trailing_pe  = info.get("trailingPE")

    # Best fallback: read from the already-computed PE cache (no extra API call)
    if not trailing_pe or not trailing_eps:
        pe_cached = cache_get("pe", 7200)
        if pe_cached:
            if not trailing_pe:
                trailing_pe = pe_cached.get("current_pe")
            if not trailing_eps and trailing_pe and fi_data.get("price"):
                trailing_eps = round(fi_data["price"] / trailing_pe, 2)
            if trailing_pe:
                log.info("EPS/PE from PE cache: pe=%.2f eps=%.2f",
                         trailing_pe or 0, trailing_eps or 0)

    # Last resort: compute EPS from earnings_history + exchange rate
    if not trailing_eps:
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty and "epsActual" in eh.columns:
                ttm_kzt = float(eh["epsActual"].tail(4).sum())
                if ttm_kzt > 0:
                    usd_kzt = _get_usd_kzt_rate()
                    trailing_eps = round(ttm_kzt / usd_kzt, 2)
                    log.info("Trailing EPS from earnings_history: $%.2f", trailing_eps)
        except Exception as e:
            log.warning("EPS earnings_history fallback failed: %s", e)

    if not trailing_pe and trailing_eps and fi_data.get("price"):
        trailing_pe = round(fi_data["price"] / trailing_eps, 2)
        log.info("Trailing PE computed from price/eps: %.2f", trailing_pe)

    mc = info.get("marketCap") or fi_data.get("market_cap")

    return {
        "name":           info.get("longName", COMPANY_NAME),
        "exchange":       info.get("exchange", "NASDAQ"),
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
# Historical P/E  (KSPI-specific: KZT financials -> USD P/E)
# ---------------------------------------------------------------------------

def _get_usd_kzt_rate() -> float:
    """Fetch live USD/KZT exchange rate from yfinance. Returns KZT per 1 USD."""
    try:
        rate = float(yf.Ticker("USDKZT=X").fast_info.last_price)
        if 300 < rate < 1000:  # sanity check
            log.info("USDKZT=X rate: %.2f", rate)
            return rate
    except Exception as e:
        log.warning("USDKZT=X fetch failed: %s", e)
    return 463.0  # approximate fallback (~2024–2025 average)


def _build_ttm_eps_steps(t: yf.Ticker, info: dict) -> list:
    """
    Return list of (date_str, ttm_eps_usd) sorted ascending.
    Each entry is the TTM EPS valid from that date forward.
    Works even if info is empty (cloud-safe fallbacks).
    """
    trailing_eps_usd = info.get("trailingEps")

    # Shares: prefer info, fall back to fast_info
    shares = (info.get("sharesOutstanding") or
              info.get("impliedSharesOutstanding") or 0)
    if not shares:
        try:
            shares = int(t.fast_info.shares)
            log.info("Shares from fast_info: %d", shares)
        except Exception as e:
            log.warning("fast_info.shares failed: %s", e)

    steps = []

    # --- Determine KZT->USD conversion rate ---
    kzt_to_usd = None

    # Method 1: trailingEps (USD) / sum(last 4 quarters KZT EPS)
    if trailing_eps_usd:
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty and "epsActual" in eh.columns:
                last4 = float(eh["epsActual"].tail(4).sum())
                if last4 > 0:
                    kzt_to_usd = float(trailing_eps_usd) / last4
                    log.info("KZT->USD = %.6f (earnings_history)", kzt_to_usd)
        except Exception as e:
            log.debug("earnings_history kzt_to_usd: %s", e)

    # Method 2: trailingEps (USD) / annual NI per share (KZT)
    if kzt_to_usd is None and trailing_eps_usd and shares:
        try:
            ist = t.income_stmt
            if ist is not None and "Net Income" in ist.index:
                ni_kzt = float(ist.loc["Net Income"].dropna().iloc[0])
                eps_kzt = ni_kzt / shares
                kzt_to_usd = float(trailing_eps_usd) / eps_kzt
                log.info("KZT->USD = %.6f (income_stmt fallback)", kzt_to_usd)
        except Exception as e:
            log.debug("income_stmt kzt_to_usd: %s", e)

    # Method 3: live USDKZT exchange rate (works without t.info)
    if kzt_to_usd is None:
        usd_kzt = _get_usd_kzt_rate()
        kzt_to_usd = 1.0 / usd_kzt
        log.info("KZT->USD = %.6f (exchange rate: 1/%.0f)", kzt_to_usd, usd_kzt)

    if not shares:
        log.error("Cannot determine shares outstanding")
        return []

    # --- Collect quarterly KZT EPS per share ---
    qtly = {}  # date_str -> kzt_eps_per_share

    # From earnings_history (actual reported)
    try:
        eh = t.earnings_history
        if eh is not None and not eh.empty and "epsActual" in eh.columns:
            for dt, row in eh.iterrows():
                qtly[str(dt)[:10]] = float(row["epsActual"])
    except Exception:
        pass

    # From quarterly_financials (Net Income / shares)
    try:
        qf = t.quarterly_financials
        if qf is not None and "Net Income" in qf.index:
            for dt, ni in qf.loc["Net Income"].dropna().items():
                d = str(dt)[:10]
                if d not in qtly:
                    qtly[d] = float(ni) / shares
    except Exception as e:
        log.debug("quarterly_financials: %s", e)

    # --- Annual KZT EPS from income_stmt ---
    annual = {}  # year_int -> kzt_eps_per_share
    try:
        ist = t.income_stmt
        if ist is not None and "Net Income" in ist.index:
            for dt, ni in ist.loc["Net Income"].dropna().items():
                yr = int(str(dt)[:4])
                annual[yr] = float(ni) / shares
    except Exception as e:
        log.debug("income_stmt annual: %s", e)

    # --- Backfill 2024 quarterly split from annual total ---
    q4_2024_kzt = qtly.get("2024-12-31", 0)
    if 2024 in annual and q4_2024_kzt > 0:
        remainder = annual[2024] - q4_2024_kzt
        if remainder > 0:
            per_q = remainder / 3
            for qkey in ["2024-03-31", "2024-06-30", "2024-09-30"]:
                if qkey not in qtly:
                    qtly[qkey] = per_q
            log.info("Estimated Q1-Q3 2024 EPS (~%.0f KZT/q)", per_q)

    # --- Compute TTM EPS from sorted quarters ---
    sorted_qtly = sorted(qtly.items())
    if len(sorted_qtly) >= 4:
        dates = [x[0] for x in sorted_qtly]
        vals  = [x[1] for x in sorted_qtly]
        for i in range(3, len(sorted_qtly)):
            ttm_kzt = sum(vals[i-3:i+1])
            ttm_usd = ttm_kzt * kzt_to_usd
            if ttm_usd > 0:
                steps.append((dates[i], round(ttm_usd, 4)))
        log.info("Built %d TTM steps from quarterly data", len(steps))

    # --- Annual anchors for periods before quarterly coverage ---
    first_step = steps[0][0] if steps else "9999-12-31"
    for yr in sorted(annual.keys()):
        yr_str = f"{yr}-12-31"
        usd    = annual[yr] * kzt_to_usd
        if yr_str < first_step and usd > 0:
            steps.append((yr_str, round(usd, 4)))
            log.info("Annual anchor %s: TTM EPS=$%.2f", yr_str, usd)

    steps.sort(key=lambda x: x[0])
    return steps


def fetch_historical_pe() -> dict | None:
    t = _ticker()

    # t.info can fail on cloud servers (Yahoo Finance IP blocks) — always use try/except
    info = {}
    try:
        raw = t.info
        if raw and len(raw) > 5:
            info = raw
            log.info("t.info OK (%d fields)", len(info))
        else:
            log.warning("t.info returned empty/minimal dict")
    except Exception as e:
        log.warning("t.info failed: %s", e)

    current_pe    = info.get("trailingPE")
    current_eps   = info.get("trailingEps")
    current_price = info.get("regularMarketPrice") or info.get("currentPrice")

    # Fallback: get current price from fast_info if info is empty
    if not current_price:
        try:
            current_price = float(t.fast_info.last_price)
            log.info("current_price from fast_info: $%.2f", current_price)
        except Exception as e:
            log.warning("fast_info.last_price failed: %s", e)

    # Build EPS step timeline
    eps_steps = _build_ttm_eps_steps(t, info)

    if not eps_steps:
        if current_pe and current_pe > 0:
            log.warning("Returning current P/E only (no history)")
            return {
                "pe_history": [], "current_pe": round(float(current_pe), 2),
                "val_label": "unbekannt", "val_color": "#94a3b8",
                "stats": {}, "eps_source": "current_only",
                "timestamp": datetime.now().isoformat(),
            }
        return None

    # Full daily price history
    hist = _ticker().history(period="max", interval="1d")
    hist.index = _tz_strip(hist.index)
    hist = hist.sort_index()

    if hist.empty:
        log.error("Empty price history")
        return None

    # Apply step-function EPS to each trading day
    pe_rows = []
    step_idx = 0

    for date, row in hist.iterrows():
        date_str = str(date)[:10]
        # Advance to latest applicable step
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
        if 0 < pe < 300:
            pe_rows.append({
                "date":    date_str,
                "pe":      round(pe, 2),
                "price":   round(price, 2),
                "ttm_eps": round(ttm_eps, 4),
            })

    # If info was empty, compute current PE from the latest EPS step + live price
    if not current_pe and eps_steps and current_price:
        latest_eps = eps_steps[-1][1]
        if latest_eps > 0:
            current_pe  = round(current_price / latest_eps, 2)
            current_eps = latest_eps
            log.info("Computed current PE=%.2f from eps_steps (no info)", current_pe)

    # Ensure today is represented
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
        log.error("No P/E rows computed")
        return None

    vals = [r["pe"] for r in pe_rows]
    pcts = np.percentile(vals, [10, 25, 50, 75, 90])

    # Valuation label
    val_label = "unbekannt"
    val_color = "#94a3b8"
    if current_pe:
        cp = float(current_pe)
        if   cp <= pcts[1]: val_label, val_color = "Historisch guenstig", "#22c55e"
        elif cp <= pcts[2]: val_label, val_color = "Leicht guenstig",     "#86efac"
        elif cp <= pcts[3]: val_label, val_color = "Fair bewertet",       "#eab308"
        elif cp <= pcts[4]: val_label, val_color = "Leicht teuer",        "#f97316"
        else:               val_label, val_color = "Historisch teuer",    "#ef4444"

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
        "eps_source":  "daily_step_function",
        "timestamp":   datetime.now().isoformat(),
    }


def fetch_price_history() -> list:
    try:
        hist = _ticker().history(period="2y", interval="1d")
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
        log.error("Price history failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", ticker=TICKER, company=COMPANY_NAME)


# Service worker must be served from root scope
@app.route("/sw.js")
def service_worker():
    resp = send_from_directory("static", "sw.js")
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


# Web-app manifest
@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json")


@app.route("/api/live")
def api_live():
    data = cache_get("live", 30)
    if data is None:
        data = fetch_live_price()
        if data:
            cache_set("live", data)
    if data:
        return jsonify(data)
    return jsonify({"error": "Live-Daten nicht verfuegbar"}), 503


@app.route("/api/info")
def api_info():
    data = cache_get("info", 3600)
    if data is None:
        data = fetch_company_info()
        if data:
            cache_set("info", data)
    return jsonify(data or {"error": "Unternehmensinfo nicht verfuegbar"})


@app.route("/api/pe")
def api_pe():
    data = cache_get("pe", 3600)
    if data is None:
        log.info("Berechne historisches KGV ...")
        data = fetch_historical_pe()
        if data:
            cache_set("pe", data)
    if data:
        return jsonify(data)
    return jsonify({"error": "KGV-Daten nicht verfuegbar"}), 503


@app.route("/api/price-history")
def api_price_history():
    data = cache_get("ph", 3600)
    if data is None:
        data = fetch_price_history()
        if data:
            cache_set("ph", data)
    return jsonify(data or [])


@app.route("/api/health")
def api_health():
    """Health-check endpoint for Render.com."""
    return jsonify({"status": "ok", "ticker": TICKER,
                    "cached_keys": list(_cache.keys())})


# ---------------------------------------------------------------------------
# Background warm-up  (pre-loads data so first request is fast)
# ---------------------------------------------------------------------------

def _warmup():
    """Called in a daemon thread at startup – pre-fetches all heavy data."""
    import time as _time
    _time.sleep(2)  # Let Flask fully start first
    log.info("Warmup: pre-loading live price ...")
    d = fetch_live_price()
    if d:
        cache_set("live", d)
    # PE first — company info reads EPS/PE from the PE cache as a fallback
    log.info("Warmup: pre-loading historical P/E (this may take ~20s) ...")
    d = fetch_historical_pe()
    if d:
        cache_set("pe", d)
    log.info("Warmup: pre-loading company info ...")
    d = fetch_company_info()
    if d:
        cache_set("info", d)
    log.info("Warmup: pre-loading price history ...")
    d = fetch_price_history()
    if d:
        cache_set("ph", d)
    log.info("Warmup complete.")


threading.Thread(target=_warmup, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    local_ip = _local_ip()
    port = int(__import__("os").environ.get("PORT", 5000))
    print("\n" + "=" * 55)
    print("  Kaspi.kz (KSPI) Stock Dashboard")
    print("=" * 55)
    print(f"  PC-Browser:   http://localhost:{port}")
    print(f"  Handy (PWA):  http://{local_ip}:{port}")
    print()
    print("  -> Handy muss im selben WLAN sein")
    print("  -> Android: Chrome-Menu -> 'Zum Startbildschirm'")
    print("  -> iPhone:  Safari-Teilen -> 'Zum Home-Bildschirm'")
    print("=" * 55 + "\n")
    app.run(debug=False, port=port, host="0.0.0.0")


