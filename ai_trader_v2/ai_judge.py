"""
Autonomous AI Trade Judge (`ai_trader_v2/ai_judge.py`):
- Reads multi-TF candle picture (15m / 1H / 4H / BTC) + attached in-memory PNG candlestick chart,
  all fired TA schools (out of 12), OI/Funding derivatives, orderbook depth/imbalance, S/R obstacles,
  and open portfolio state.
- Supports configurable provider selection ("anthropic" | "gemini" | "openai" | "auto") and explicit
  model override from `config_ai_trader.json` without editing `.env`.
- Accurate per-model USD cost tracking in `data/ai_trader_v2/ai_usage.json` + hard Python guardrails.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from ai_trader_v2.ta_schools import CandidateSetup

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
AI_USAGE_PATH = Path(os.getenv("AI_TRADER_DATA_DIR", str(ROOT_DIR / "data" / "ai_trader_v2"))) / "ai_usage.json"

_SECRET_RE = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{10,}|AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{16,}|\b\d{8,12}:[A-Za-z0-9_\-]{25,}\b)"
)


def redact_secrets(text: Any) -> str:
    return _SECRET_RE.sub("[REDACTED]", str(text))


def get_model_rates_per_m(provider: str, model: str) -> Tuple[float, float]:
    """Return accurate (input_usd_per_1M, output_usd_per_1M) pricing for the active model."""
    p = provider.lower().strip()
    m = model.lower().strip()
    if p == "anthropic":
        if "opus" in m:
            # Opus 4.5/5 tier ($5/$25); legacy Opus 4/4.1 ($15/$75)
            if any(k in m for k in ("opus-4-1", "opus-4-2025", "opus-3")):
                return 15.0, 75.0
            return 5.0, 25.0
        if "haiku" in m:
            return 0.80, 4.0
        return 3.0, 15.0  # Sonnet default
    if p == "gemini":
        if "flash-lite" in m:
            return 0.075, 0.30
        if "pro" in m:
            return 1.25, 10.0
        return 0.15, 0.60  # Gemini Flash default
    # OpenAI-compatible
    if "mini" in m or "nano" in m:
        return 0.15, 0.60
    if "gpt-4o" in m or "gpt-4.1" in m:
        return 2.50, 10.0
    return 1.0, 4.0


TRADER_SYSTEM_PROMPT = """Ты — старший риск-аналитик и дискреционный трейдер по криптофьючерсам Bybit USDT Perpetual (таймфрейм 15M, плечо 10x, маржа 10% от депозита, до 8 позиций / до 5 в одну сторону).

Твоя ЕДИНСТВЕННАЯ задача — изучить приложенный PNG-график 15M (если передан) и ВСЮ картину свечей в JSON (последние 20 свечей 15M, 12 свечей 1H, 8 свечей 4H, свечи BTCUSDT), сработавшие школы ТА (из 12), деривативы (OI, Funding Rate), стакан, ближайшие уровни 1H/4H и открытые позиции — и решить:
ВХОДИТЬ (ENTER) или ПРОПУСТИТЬ (SKIP).

ПРАВИЛА ОТБОРА (СТРОГО):
1. Защита капитала важнее количества сделок. На 10x плече 1 плохая сделка стоит -1.5% депозита. Лучше отклонить 60% слабых сигналов и пропустить только чистые сетапы.
2. ВСЕГДА проверяй визуальную картину графика и свечей:
   - Для LONG требуется подтверждение 1H + 4H; для SHORT допускаются шорты по слому 1H (даже если 4H ещё BULL/NEUTRAL) — внимательно оценивай контекст 4H, чтобы не шортить прямо в мощную поддержку 4H!
   - ОТКЛОНЯЙ (SKIP), если вход идёт прямо в сильное сопротивление (LONG) или поддержку (SHORT) на 1H/4H и запас хода меньше 1.5R (`nearest_obstacle_in_R < 1.5`).
   - ОТКЛОНЯЙ (SKIP), если последняя свеча 15M — аномальный вертикальный выстрел (тело > 2.2 ATR) без отката: входить на самом экстремуме нельзя.
   - ОТКЛОНЯЙ (SKIP), если на графике 15M видна серия длинных теней против входа (отторжение уровня, shooting_pin против LONG / hammer_pin против SHORT).
   - ОТКЛОНЯЙ (SKIP), если BTC прямо сейчас рисует сильные разворотные свечи против нашей сделки.
   - ОТКЛОНЯЙ (SKIP), если в портфеле уже открыто 3–4 сильно коррелированных альта в ту же сторону и новый сигнал слабый.
3. МУЛЬТИ-ИНДИКАТОРНЫЕ ДИВЕРГЕНЦИИ (`setup.divergences` по RSI/MACD_HIST/OBV/MFI на 15M, 1H, 4H):
   - `4H` дивергенция — это макро-предупреждение об излёте основного тренда. Если `h4_exhaustion_warning == true` и на 4H горит сильная встречная REGULAR дивергенция (`3–4/4`, `strength="STRONG"`), относись к входу максимально осторожно и чаще выбирай SKIP, если нет явного пробоя на мощном объёме.
   - `1H` и `15m` дивергенции — подтверждение структуры и тайминга входа: совпадение `confirms_15m` и/или `confirms_1h` (обычная или скрытая дивергенция в сторону сделки) повышает уверенность (`confidence`).
   - Отсутствие дивергенций (`NONE`) в чистом импульсном тренде НЕ является причиной для SKIP!
4. СОВЕТНИКИ TRADINGVIEW (`setup.tv_advisors`: `liquidity_sweep`, `volume_profile`, `session_levels`):
   - Все 3 советника строго информационные: отсутствие SFP (`sweep_type == "NONE"`) НЕ является причиной для SKIP в чистом тренде.
   - Встречный SFP (`trap_warning == true`) или вход вплотную в `POC` (`poc_obstacle_ahead == true`) либо в `PDH`/`PDL` (`session_obstacle_ahead == true`) — сильное предупреждение о ловушке или ближайшем сопротивлении (повод снизить уверенность или выбрать SKIP).
   - Попутный SFP (`aligns_with_direction == true`, особенно с `with_volume == true`) и свободное пространство над `VAH` (`ABOVE_VAH` для LONG) или под `VAL` (`BELOW_VAL` для SHORT) — весомое подтверждение для ENTER.
5. ПРИОРИТЕТНЫЕ СЕТАПЫ (высокий шанс на ENTER):
   - Сработала `S12_Deriv_OI_Funding` с подтверждённым объёмом (`rvol >= 1.2`) при чистой структуре свечей.
   - Совпадение 2+ независимых школ ТА (`confluence_count >= 2`) + объём (`rvol >= 1.2`) + свободный запас хода `>= 2.0R`.
6. ПОСТАНОВКА SL, TP и 3-СТУПЕНЧАТОЕ СОПРОВОЖДЕНИЕ:
   - Выбирай `sl_pct` строго внутри `allowed_sl_corridor_pct` (1.2%–1.6%) и не ниже `atr_floor_sl_pct`.
   - Ставь `tp1_r` в диапазоне `1.5–2.2` (обычно `1.8`). Умное сопровождение работает в 3 ступени: закрытие 15M >= +1.3R -> SL -0.5R; после TP1 (55%) -> SL остатка в буфер -0.35R (даём дышать на ретесте); закрытие 15M >= +2.3R -> SL в плюс +0.3R + ATR-трейлинг.
   - Ставь `tp2_r` в диапазоне `2.6–4.5`, либо включи `use_trailing_after_tp1: true` для трендовых пробоев.

ФОРМАТ ОТВЕТА — ТОЛЬКО ВАЛИДНЫЙ JSON БЕЗ ЛИШНЕГО ТЕКСТА:
{
  "decision": "ENTER" или "SKIP",
  "confidence": целое число от 0 до 100,
  "selected_school": "название главной школы из fired_schools",
  "sl_pct": число от 1.2 до 1.6,
  "tp1_r": число от 1.5 до 2.5,
  "tp2_r": число от 2.5 до 5.0,
  "use_trailing_after_tp1": true или false,
  "reason_ru": "Краткий разбор на русском (макс 2 предложения, до 260 символов: график/свечи 15M/1H/4H, уровни, BTC, OI/Funding и итог)",
  "risk_warnings": ["короткий риск 1 (до 60 символов)", "короткий риск 2"]
}
"""


@dataclass
class AIDecision:
    decision: str  # "ENTER" or "SKIP"
    confidence: int
    selected_school: str
    sl_pct: float
    tp1_r: float
    tp2_r: float
    use_trailing_after_tp1: bool
    reason_ru: str
    risk_warnings: List[str] = field(default_factory=list)
    provider: str = ""
    raw_approved: bool = False
    guardrail_notes: List[str] = field(default_factory=list)


class AITradeJudge:
    """Multi-provider multimodal LLM trade judge with explicit config selection & accurate USD billing."""

    def __init__(
        self,
        monthly_limit_usd: float = 50.0,
        provider: str = "auto",
        model: str = "",
    ) -> None:
        self.monthly_limit_usd = monthly_limit_usd
        self.config_provider = (provider or "auto").lower().strip()
        self.config_model = (model or "").strip()

        self.anthropic_key = ""
        self.anthropic_model = "claude-sonnet-4-20250514"
        self.gemini_key = ""
        self.gemini_model = "gemini-2.5-flash"
        self.openai_key = ""
        self.openai_base = "https://api.openai.com/v1"
        self.openai_model = "gpt-4o-mini"
        self.reload_env()

    def reload_env(self) -> None:
        load_dotenv(override=True)
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        env_ant_model = os.getenv("AI_TRADER_ANTHROPIC_MODEL", "").strip() or os.getenv("ANTHROPIC_MODEL", "").strip()
        if env_ant_model:
            self.anthropic_model = env_ant_model
        if self.config_provider == "anthropic" and self.config_model:
            self.anthropic_model = self.config_model

        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        g_model = os.getenv("GEMINI_MODEL", "").strip()
        if g_model:
            self.gemini_model = g_model
        if self.config_provider == "gemini" and self.config_model:
            self.gemini_model = self.config_model

        self.openai_key = os.getenv("AI_API_KEY", "").strip()
        o_base = os.getenv("AI_API_BASE_URL", "").strip()
        if o_base:
            self.openai_base = o_base.rstrip("/")
        o_model = os.getenv("AI_MODEL", "").strip()
        if o_model:
            self.openai_model = o_model
        if self.config_provider == "openai" and self.config_model:
            self.openai_model = self.config_model

        env_limit = os.getenv("AI_MONTHLY_LIMIT_USD", "").strip()
        if env_limit:
            try:
                self.monthly_limit_usd = max(1.0, float(env_limit))
            except ValueError:
                pass

    def _has_key(self, provider: str) -> bool:
        k = {"anthropic": self.anthropic_key, "gemini": self.gemini_key, "openai": self.openai_key}.get(provider, "")
        return bool(k and "your_" not in k.lower())

    def _provider_order(self) -> List[str]:
        all_p = ["anthropic", "gemini", "openai"]
        if self.config_provider in all_p:
            return [self.config_provider] + [p for p in all_p if p != self.config_provider]
        return all_p

    def _model_for(self, provider: str) -> str:
        if provider == "anthropic":
            return self.anthropic_model
        if provider == "gemini":
            return self.gemini_model
        return self.openai_model

    @property
    def is_configured(self) -> bool:
        return any(self._has_key(p) for p in self._provider_order())

    @property
    def active_provider_label(self) -> str:
        for p in self._provider_order():
            if self._has_key(p):
                m = self._model_for(p)
                in_r, out_r = get_model_rates_per_m(p, m)
                return f"{p.capitalize()} ({m}, ${in_r:g}/${out_r:g} per 1M)"
        return "NONE"

    def _load_usage(self) -> Dict[str, Any]:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        if AI_USAGE_PATH.exists():
            try:
                data = json.loads(AI_USAGE_PATH.read_text(encoding="utf-8"))
                if data.get("month") == month:
                    return data
            except Exception:
                pass
        return {"month": month, "spent_usd": 0.0, "requests": 0, "input_tokens": 0, "output_tokens": 0}

    def _record_usage(self, provider: str, model: str, in_tok: int, out_tok: int) -> None:
        in_rate, out_rate = get_model_rates_per_m(provider, model)
        usage = self._load_usage()
        cost = (in_tok / 1_000_000.0) * in_rate + (out_tok / 1_000_000.0) * out_rate
        usage["input_tokens"] = int(usage.get("input_tokens", 0)) + in_tok
        usage["output_tokens"] = int(usage.get("output_tokens", 0)) + out_tok
        usage["requests"] = int(usage.get("requests", 0)) + 1
        usage["spent_usd"] = round(float(usage.get("spent_usd", 0.0)) + cost, 6)
        usage["last_provider"] = f"{provider}:{model}"
        AI_USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        AI_USAGE_PATH.write_text(json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_budget_status(self) -> Dict[str, Any]:
        u = self._load_usage()
        spent = float(u.get("spent_usd", 0.0))
        return {
            "month": u.get("month"),
            "spent_usd": round(spent, 4),
            "limit_usd": self.monthly_limit_usd,
            "remaining_usd": round(max(0.0, self.monthly_limit_usd - spent), 4),
            "requests": int(u.get("requests", 0)),
            "input_tokens": int(u.get("input_tokens", 0)),
            "output_tokens": int(u.get("output_tokens", 0)),
        }

    def _call_anthropic(self, sys_txt: str, user_text: str, max_tokens: int, image_png_b64: Optional[str]) -> Optional[str]:
        content: Any
        if image_png_b64:
            content = [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": image_png_b64},
                },
                {"type": "text", "text": user_text},
            ]
        else:
            content = user_text

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.anthropic_model,
                "max_tokens": max_tokens,
                "system": sys_txt,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=(10, 45),
        )
        if resp.status_code == 200:
            data = resp.json()
            u = data.get("usage", {})
            self._record_usage("anthropic", self.anthropic_model, int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)))
            blocks = data.get("content", [])
            txt = "\n".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text").strip()
            return txt or None
        logger.warning("Anthropic error %s: %s", resp.status_code, redact_secrets(resp.text[:200]))
        return None

    def _call_gemini(self, sys_txt: str, user_text: str, max_tokens: int, image_png_b64: Optional[str]) -> Optional[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        parts: List[Dict[str, Any]] = []
        if image_png_b64:
            parts.append({"inline_data": {"mime_type": "image/png", "data": image_png_b64}})
        parts.append({"text": user_text})

        resp = requests.post(
            url,
            headers={"x-goog-api-key": self.gemini_key, "Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": sys_txt}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0.15, "maxOutputTokens": max_tokens},
            },
            timeout=(10, 45),
        )
        if resp.status_code == 200:
            data = resp.json()
            um = data.get("usageMetadata", {})
            self._record_usage(
                "gemini", self.gemini_model, int(um.get("promptTokenCount", 0)), int(um.get("candidatesTokenCount", 0))
            )
            cands = data.get("candidates", [])
            if cands:
                out_parts = cands[0].get("content", {}).get("parts", [])
                txt = "\n".join(str(p.get("text", "")) for p in out_parts if "text" in p).strip()
                return txt or None
        logger.warning("Gemini error %s: %s", resp.status_code, redact_secrets(resp.text[:200]))
        return None

    def _call_openai(self, sys_txt: str, user_text: str, max_tokens: int, image_png_b64: Optional[str]) -> Optional[str]:
        if image_png_b64:
            user_content: Any = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_png_b64}"}},
                {"type": "text", "text": user_text},
            ]
        else:
            user_content = user_text

        resp = requests.post(
            f"{self.openai_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
            json={
                "model": self.openai_model,
                "temperature": 0.15,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": sys_txt},
                    {"role": "user", "content": user_content},
                ],
            },
            timeout=(10, 45),
        )
        if resp.status_code == 200:
            data = resp.json()
            u = data.get("usage", {})
            self._record_usage("openai", self.openai_model, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))
            txt = data["choices"][0]["message"]["content"].strip()
            return txt or None
        logger.warning("OpenAI error %s: %s", resp.status_code, redact_secrets(resp.text[:200]))
        return None

    def _call_llm(
        self,
        user_text: str,
        max_tokens: int = 1600,
        system_prompt: Optional[str] = None,
        image_png_b64: Optional[str] = None,
    ) -> tuple[Optional[str], str]:
        self.reload_env()
        if not self.is_configured:
            return None, "unconfigured"
        usage = self._load_usage()
        if float(usage.get("spent_usd", 0.0)) >= self.monthly_limit_usd:
            logger.warning("AI monthly budget limit $%.2f reached — skipping AI call.", self.monthly_limit_usd)
            return None, "budget_exceeded"
        sys_txt = system_prompt or TRADER_SYSTEM_PROMPT

        dispatch = {
            "anthropic": self._call_anthropic,
            "gemini": self._call_gemini,
            "openai": self._call_openai,
        }
        for prov in self._provider_order():
            if not self._has_key(prov):
                continue
            fn = dispatch[prov]
            model_name = self._model_for(prov)
            try:
                txt = fn(sys_txt, user_text, max_tokens, image_png_b64)
                if txt:
                    return txt, f"{prov}:{model_name}"
                # Safe fallback: if multimodal PNG call failed, retry text-only once
                if image_png_b64:
                    logger.info("Retrying %s text-only fallback without PNG chart.", prov)
                    txt2 = fn(sys_txt, user_text, max_tokens, None)
                    if txt2:
                        return txt2, f"{prov}:{model_name}"
            except Exception as exc:
                logger.warning("%s request failed: %s", prov.capitalize(), redact_secrets(exc))

        return None, "failed"

    @staticmethod
    def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
        txt = raw_text.strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    def evaluate_candidate(
        self,
        candidate: CandidateSetup,
        portfolio_state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> AIDecision:
        """Send full candle + PNG chart + TA context to AI Judge and enforce hard Python guardrails."""
        payload = {
            "setup": candidate.context_payload,
            "portfolio": portfolio_state,
        }
        img_note = "Приложен PNG-график последних закрытых свечей 15M (EMA20/50, VWAP24h, уровни 1H/4H, объём).\n" if candidate.chart_png_b64 else ""
        prompt = (
            f"Оцени кандидата {candidate.symbol} ({candidate.direction}) на закрытой свече 15M.\n"
            f"{img_note}"
            f"Сработавшие школы ({len(candidate.fired_schools)}): {', '.join(candidate.fired_schools)}.\n"
            f"Верни ТОЛЬКО строгий JSON по заданной схеме.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        raw_txt, provider = self._call_llm(prompt, image_png_b64=candidate.chart_png_b64)
        if not raw_txt:
            return AIDecision(
                decision="SKIP",
                confidence=0,
                selected_school=candidate.primary_school,
                sl_pct=candidate.default_sl_pct,
                tp1_r=candidate.recommended_tp1_r,
                tp2_r=candidate.recommended_tp2_r,
                use_trailing_after_tp1=False,
                reason_ru=f"AI-судья недоступен ({provider}) — сработала защита fail-closed, вход пропущен.",
                risk_warnings=[f"ai_unavailable:{provider}"],
                provider=provider,
                raw_approved=False,
                guardrail_notes=["fail_closed_ai_unavailable"],
            )

        parsed = self._extract_json(raw_txt)
        if not parsed:
            return AIDecision(
                decision="SKIP",
                confidence=0,
                selected_school=candidate.primary_school,
                sl_pct=candidate.default_sl_pct,
                tp1_r=candidate.recommended_tp1_r,
                tp2_r=candidate.recommended_tp2_r,
                use_trailing_after_tp1=False,
                reason_ru="AI вернул невалидный JSON — защита fail-closed отклонила вход.",
                risk_warnings=["invalid_json_response"],
                provider=provider,
                raw_approved=False,
                guardrail_notes=["fail_closed_invalid_json"],
            )

        return self.validate_and_clamp(parsed, candidate, config, provider)

    @staticmethod
    def validate_and_clamp(
        parsed: Dict[str, Any],
        candidate: CandidateSetup,
        config: Dict[str, Any],
        provider: str,
    ) -> AIDecision:
        """Hard Python safety validator — LLM cannot violate risk, SL/TP, or confidence rules."""
        notes: List[str] = []
        raw_dec = str(parsed.get("decision", "SKIP")).upper().strip()
        raw_approved = raw_dec == "ENTER"

        try:
            conf = int(round(float(parsed.get("confidence", 0))))
        except (TypeError, ValueError):
            conf = 0
        conf = max(0, min(100, conf))

        sel_school = str(parsed.get("selected_school") or candidate.primary_school).strip()
        if sel_school not in candidate.fired_schools:
            sel_school = candidate.primary_school

        # 1. Hard SL corridor [sl_min_pct, sl_max_pct] AND >= sl_atr_floor_mult * atr_pct
        sl_min = float(config.get("sl_min_pct", 1.2))
        sl_max = float(config.get("sl_max_pct", 1.6))
        atr_floor = float(config.get("sl_atr_floor_mult", 1.1)) * candidate.atr_pct
        effective_min = max(sl_min, atr_floor)

        try:
            sl_pct = float(parsed.get("sl_pct", candidate.default_sl_pct))
        except (TypeError, ValueError):
            sl_pct = candidate.default_sl_pct

        if sl_pct < effective_min or sl_pct > sl_max:
            notes.append(f"SL clamped from {sl_pct:.2f}% to [{effective_min:.2f}%, {sl_max:.2f}%]")
            sl_pct = min(sl_max, max(effective_min, sl_pct))
        sl_pct = round(sl_pct, 3)

        # 2. Hard TP1 & TP2 R-multiple validation
        min_tp1 = float(config.get("min_rr_tp1", 1.5))
        max_tp1 = float(config.get("max_rr_tp1", 2.5))
        min_tp2 = float(config.get("min_rr_tp2", 2.5))
        max_tp2 = float(config.get("max_rr_tp2", 5.0))

        try:
            tp1_r = float(parsed.get("tp1_r", candidate.recommended_tp1_r))
        except (TypeError, ValueError):
            tp1_r = candidate.recommended_tp1_r
        tp1_r = round(min(max_tp1, max(min_tp1, tp1_r)), 2)

        try:
            tp2_r = float(parsed.get("tp2_r", candidate.recommended_tp2_r))
        except (TypeError, ValueError):
            tp2_r = candidate.recommended_tp2_r
        tp2_r = round(min(max_tp2, max(min_tp2, tp1_r + 0.6, tp2_r)), 2)

        use_trail = bool(parsed.get("use_trailing_after_tp1", False))
        reason_ru = str(parsed.get("reason_ru", "")).strip() or "Без комментария AI."
        warnings = [str(w) for w in (parsed.get("risk_warnings") or []) if str(w).strip()][:5]

        # 3. Hard Risk-per-Trade check (trading-core.md: <= 2.5% incl. fees)
        leverage = float(config.get("leverage", 10))
        margin_pct = float(config.get("margin_pct_per_trade", 10.0))
        max_risk_pct = float(config.get("max_risk_pct_per_trade", 2.5))
        round_trip_fee_pct = (float(config.get("maker_fee_rate", 0.0002)) + float(config.get("taker_fee_rate", 0.00055))) * 100.0
        est_risk_pct = (margin_pct * leverage / 100.0) * (sl_pct + round_trip_fee_pct)

        final_dec = "ENTER" if raw_approved else "SKIP"
        min_conf = int(config.get("min_ai_confidence", 72))
        if final_dec == "ENTER" and conf < min_conf:
            final_dec = "SKIP"
            notes.append(f"Rejected by guardrail: confidence {conf} < {min_conf}")
        if final_dec == "ENTER" and est_risk_pct > max_risk_pct:
            final_dec = "SKIP"
            notes.append(f"Rejected by guardrail: est risk {est_risk_pct:.2f}% > {max_risk_pct:.2f}%")
        if final_dec == "ENTER" and effective_min > sl_max:
            final_dec = "SKIP"
            notes.append(f"Rejected by guardrail: ATR floor {effective_min:.2f}% > SL max {sl_max:.2f}%")

        return AIDecision(
            decision=final_dec,
            confidence=conf,
            selected_school=sel_school,
            sl_pct=sl_pct,
            tp1_r=tp1_r,
            tp2_r=tp2_r,
            use_trailing_after_tp1=use_trail,
            reason_ru=reason_ru,
            risk_warnings=warnings,
            provider=provider,
            raw_approved=raw_approved,
            guardrail_notes=notes,
        )
