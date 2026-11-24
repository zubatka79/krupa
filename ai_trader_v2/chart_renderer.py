"""
Headless In-Memory PNG Chart Renderer for AI-Trader v2 (`ai_trader_v2/chart_renderer.py`).
Renders the last 60-80 closed 15m candles with:
  - Candlesticks (green/red bodies & wicks)
  - EMA20, EMA50, VWAP24h
  - Nearest 1H & 4H Support / Resistance horizontal levels
  - Proposed SL and TP1 reference levels
  - Volume subplot with SMA20(vol)
Returns a base64-encoded PNG string strictly in memory (zero temp files on disk).
Never raises: returns None on any error for safe text-JSON fallback.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def render_15m_chart_png_b64(
    symbol: str,
    direction: str,
    df15: pd.DataFrame,
    ema20: np.ndarray,
    ema50: np.ndarray,
    vwap: np.ndarray,
    sr_1h: Dict[str, Any],
    sr_4h: Dict[str, Any],
    sl_pct: float,
    tp1_r: float,
    candles_count: int = 72,
    div_banner: Optional[str] = None,
    adv_banner: Optional[str] = None,
    tv_levels: Optional[Dict[str, Optional[float]]] = None,
) -> Optional[str]:
    """Render compact headless PNG candlestick+volume chart to base64 string (or None on error)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        n_bars = max(30, min(int(candles_count), len(df15)))
        sub = df15.tail(n_bars).reset_index(drop=True)
        e20 = np.asarray(ema20[-n_bars:], dtype=float)
        e50 = np.asarray(ema50[-n_bars:], dtype=float)
        vw = np.asarray(vwap[-n_bars:], dtype=float)

        op = sub["open"].to_numpy(float)
        hi = sub["high"].to_numpy(float)
        lo = sub["low"].to_numpy(float)
        cl = sub["close"].to_numpy(float)
        vol = sub["volume"].to_numpy(float)
        vol_sma = pd.Series(vol).rolling(20, min_periods=5).mean().to_numpy(float)
        x = np.arange(n_bars, dtype=float)

        fig, (ax_p, ax_v) = plt.subplots(
            2,
            1,
            figsize=(8.0, 5.0),
            dpi=95,
            gridspec_kw={"height_ratios": [3.5, 1.0], "hspace": 0.06},
            sharex=True,
        )
        bg = "#0f1419"
        fg = "#d7dee6"
        grid_c = "#222b36"
        fig.patch.set_facecolor(bg)
        for ax in (ax_p, ax_v):
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg, labelsize=8)
            ax.grid(True, color=grid_c, linewidth=0.5, alpha=0.7)
            for sp in ax.spines.values():
                sp.set_color("#2e3947")

        # Candlesticks + Volume bars
        w = 0.62
        for i in range(n_bars):
            bull = cl[i] >= op[i]
            col = "#26a69a" if bull else "#ef5350"
            ax_p.plot([x[i], x[i]], [lo[i], hi[i]], color=col, linewidth=0.9, zorder=2)
            body_bot = min(op[i], cl[i])
            body_h = max(abs(cl[i] - op[i]), (hi[i] - lo[i]) * 0.02 + 1e-12)
            ax_p.add_patch(Rectangle((x[i] - w / 2.0, body_bot), w, body_h, facecolor=col, edgecolor=col, linewidth=0.5, zorder=3))
            ax_v.bar(x[i], vol[i], width=w, color=col, alpha=0.65, zorder=2)

        # Overlay EMA20, EMA50, VWAP24h
        ax_p.plot(x, e20, color="#29b6f6", linewidth=1.2, label="EMA20", zorder=4)
        ax_p.plot(x, e50, color="#ffa726", linewidth=1.3, label="EMA50", zorder=4)
        ax_p.plot(x, vw, color="#ab47bc", linewidth=1.1, linestyle="--", label="VWAP24h", zorder=4)

        # Volume SMA20
        ax_v.plot(x, vol_sma, color="#ffd54f", linewidth=1.0, label="Vol SMA20", zorder=3)

        # Nearest 1H & 4H S/R horizontal lines (only if within reasonable visual range of chart)
        p_min, p_max = float(np.min(lo)), float(np.max(hi))
        p_span = max(p_max - p_min, p_max * 0.01)
        vis_lo = p_min - 0.35 * p_span
        vis_hi = p_max + 0.35 * p_span

        def _plot_level(val: Optional[float], label: str, color: str, ls: str = ":") -> None:
            if val is not None and vis_lo <= val <= vis_hi:
                ax_p.axhline(val, color=color, linestyle=ls, linewidth=0.9, alpha=0.85, zorder=1)
                ax_p.text(0.5, val, f" {label} {val:.6g}", color=color, fontsize=7, va="bottom", ha="left")

        _plot_level(sr_1h.get("nearest_resistance"), "1H Res", "#ff8a65", "--")
        _plot_level(sr_1h.get("nearest_support"), "1H Sup", "#81c784", "--")
        _plot_level(sr_4h.get("nearest_resistance"), "4H Res", "#ff5252", "-.")
        _plot_level(sr_4h.get("nearest_support"), "4H Sup", "#00e676", "-.")

        # Thin dashed POC (gold/amber) and PDH/PDL (cyan/slate) ONLY if strictly inside visible 72-bar range
        if isinstance(tv_levels, dict):
            def _plot_tv_strict(val: Optional[float], label: str, color: str) -> None:
                if val is not None and p_min <= float(val) <= p_max:
                    fv = float(val)
                    ax_p.axhline(fv, color=color, linestyle="--", linewidth=0.85, alpha=0.85, zorder=2)
                    ax_p.text(n_bars * 0.52, fv, f" {label} {fv:.6g}", color=color, fontsize=7, va="bottom", ha="center")

            _plot_tv_strict(tv_levels.get("poc"), "POC", "#ffca28")
            _plot_tv_strict(tv_levels.get("pdh"), "PDH", "#4dd0e1")
            _plot_tv_strict(tv_levels.get("pdl"), "PDL", "#90a4ae")

        # Proposed Entry, SL, TP1 reference lines
        cp = float(cl[-1])
        is_l = direction == "LONG"
        sl_p = cp * (1.0 - sl_pct / 100.0) if is_l else cp * (1.0 + sl_pct / 100.0)
        tp1_p = cp * (1.0 + (sl_pct * tp1_r) / 100.0) if is_l else cp * (1.0 - (sl_pct * tp1_r) / 100.0)
        ax_p.axhline(cp, color="#ffffff", linestyle=":", linewidth=0.8, alpha=0.6)
        ax_p.axhline(sl_p, color="#ff1744", linestyle=":", linewidth=1.0, alpha=0.9)
        ax_p.text(n_bars - 1, sl_p, f"SL -{sl_pct:.2f}% ", color="#ff5252", fontsize=7, va="bottom", ha="right")
        if vis_lo <= tp1_p <= vis_hi:
            ax_p.axhline(tp1_p, color="#00e676", linestyle=":", linewidth=1.0, alpha=0.9)
            ax_p.text(n_bars - 1, tp1_p, f"TP1 {tp1_r:.1f}R ", color="#69f0ae", fontsize=7, va="bottom", ha="right")

        title_lines = [f"{symbol} 15m ({direction}) | Last {n_bars} Closed Bars | Price {cp:.6g}"]
        if div_banner:
            title_lines.append(div_banner)
        if adv_banner:
            title_lines.append(adv_banner)
        ax_p.set_title("\n".join(title_lines), color=fg, fontsize=9.0, pad=6)
        ax_p.legend(loc="upper left", fontsize=7.5, facecolor="#161d26", edgecolor="#2e3947", labelcolor=fg, ncol=3)
        ax_p.set_xlim(-1, n_bars + 1)

        # Clean time ticks on bottom axis
        ts_arr = sub["timestamp"].to_numpy(np.int64)
        tick_idxs = np.linspace(0, n_bars - 1, 7, dtype=int)
        tick_lbls = [
            pd.to_datetime(int(ts_arr[idx]), unit="ms", utc=True).strftime("%m-%d %H:%M")
            for idx in tick_idxs
        ]
        ax_v.set_xticks(tick_idxs)
        ax_v.set_xticklabels(tick_lbls, fontsize=7.5, color=fg)
        ax_v.set_ylabel("Vol", color=fg, fontsize=7.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.warning("Chart PNG render failed for %s (falling back to text JSON): %s", symbol, exc)
        return None
