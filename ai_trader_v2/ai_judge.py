"""
Autonomous AI Trade Judge (`ai_trader_v2/ai_judge.py`):
- Reads multi-TF candle picture (15m / 1H / 4H / BTC), all fired TA schools (out of 12),
  OI/Funding derivatives, orderbook depth/imbalance, S/R obstacles, and open portfolio state.
- Outputs strict JSON verdict (ENTER / SKIP + confidence + structural SL/TP + Russian rationale).
- Enforces hard Python safety guardrails (SL in [1.2%, 1.6%] >= 1.1*ATR, risk <= 2.5%, R:R bounds)
  and monthly USD API budget cap (`ai_trader_v2/data/ai_usage.json`).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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


TRADER_SYSTEM_PROMPT = """Ты — старший риск-аналитик и дискреционный трейдер по криптофьючерсам Bybit USDT Perpetual (таймфрейм 15M со старшими фильтрами 1H и 4H, плечо 10x, маржа 10% от депозита).

Твоя ЕДИНСТВЕННАЯ задача — изучить ВСЮ картину свечей (последние 20 свечей 15M, 12 свечей 1H, 8 свечей 4H, свечи BTCUSDT), сработавшие школы ТА (из 12), деривативы (OI, Funding Rate), стакан, ближайшие уровни поддержки/сопротивления 1H/4H и открытые позиции в портфеле — и решить:
ВХОДИТЬ (ENTER) или ПРОПУСТИТЬ (SKIP).

ПРАВИЛА ОТБОРА (СТРОГО):
1. Защита капитала важнее количества сделок. На 10x плече 1 плохая сделка стоит -1.5% депозита. Лучше отклонить 60% слабых сигналов и пропустить только чистые сетапы.
2. ВСЕГДА проверяй картину свечей:
   - ОТКЛОНЯЙ (SKIP), если вход идёт прямо в сильное сопротивление (для LONG) или поддержку (для SHORT) на 1H/4H и запас хода меньше 1.5R (`nearest_obstacle_in_R < 1.5`).
   - ОТКЛОНЯЙ (SKIP), если последняя свеча 15M — это аномальный вертикальный выстрел (перекупленный/перепроданный хвост или тело > 2.2 ATR) без отката: входить на самом хае/лое нельзя.
   - ОТКЛОНЯЙ (SKIP), если на свечах 15M видно серию длинных теней против направления входа (отторжение уровня, shooting_pin против LONG / hammer_pin против SHORT).
   - ОТКЛОНЯЙ (SKIP), если BTC прямо сейчас рисует сильные разворотные свечи против направления нашей сделки.
   - ОТКЛОНЯЙ (SKIP), если в портфеле уже открыто 2–3 сильно коррелированных альта в ту же сторону и новый сигнал слабый.
3. ПРИОРИТЕТНЫЕ СЕТАПЫ (высокий шанс на ENTER):
   - Сработала школа `S12_Deriv_OI_Funding` (рост OI + перекос Funding в ловушке толпы + пробой по тренду 1H/4H) при чистой структуре свечей.
   - Совпадение 2+ независимых школ ТА (`confluence_count >= 2`) + чистый объём (`rvol >= 1.3`) + тренд 1H/4H (`STRONG_TREND` или `SQUEEZE_BREAKOUT`) + свободное пространство до уровня `>= 2.0R`.
   - Один сильный сигнал из трендовых школ (`S07_Ichimoku_Kumo`, `S06_TTM_Squeeze`, `S05_RS_BTC_Breakout`, `S04_Donchian_ADX`, `S09_Trend_Pullback_RSI`, `S11_VWAP_Reclaim`), если свечная формация 15M/1H/4H идеальна (чистый ретест/поджатие, сильное закрытие без встречного хвоста, поддержка BTC).
4. ПОСТАНОВКА SL и TP:
   - Выбирай `sl_pct` строго внутри `allowed_sl_corridor_pct` (1.2%–1.6%) и не ниже `atr_floor_sl_pct`, опираясь на структурные кандидаты (`swing_10`, `kijun`, `supertrend`).
   - Ставь `tp1_r` в диапазоне `1.5–2.2` (обычно `1.8`, чуть перед ближайшим уровнем 1H, если уровень на `1.6R–2.1R`). После TP1 (55% позиции) бот автоматически переведёт стоп в безубыток `+0.15%`.
   - Ставь `tp2_r` в диапазоне `2.6–4.5` (в сильном тренде `3.5–4.0`, в умеренном `2.8–3.2`), либо включи `use_trailing_after_tp1: true` для трендовых пробоев.

ФОРМАТ ОТВЕТА — ТОЛЬКО ВАЛИДНЫЙ JSON БЕЗ ЛИШНЕГО ТЕКСТА:
{
  "decision": "ENTER" или "SKIP",
  "confidence": целое число от 0 до 100,
  "selected_school": "название главной школы из fired_schools",
  "sl_pct": число от 1.2 до 1.6,
  "tp1_r": число от 1.5 до 2.5,
  "tp2_r": число от 2.5 до 5.0,
  "use_trailing_after_tp1": true или false,
  "reason_ru": "Краткий разбор на русском (макс 2 предложения, до 260 символов: свечи 15M/1H/4H, уровни, BTC, OI/Funding и итог)",
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
    """Multi-provider LLM trade judge (Anthropic Claude -> Google Gemini -> OpenAI) with budget cap."""

    def __init__(self, monthly_limit_usd: float = 50.0) -> None:
        self.monthly_limit_usd = monthly_limit_usd
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
        # Allow override via AI_TRADER_MODEL, else fall back to ANTHROPIC_MODEL or Sonnet
        env_model = os.getenv("AI_TRADER_ANTHROPIC_MODEL", "").strip() or os.getenv("ANTHROPIC_MODEL", "").strip()
        if env_model:
            self.anthropic_model = env_model
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        g_model = os.getenv("GEMINI_MODEL", "").strip()
        if g_model:
            self.gemini_model = g_model
        self.openai_key = os.getenv("AI_API_KEY", "").strip()
        o_base = os.getenv("AI_API_BASE_URL", "").strip()
        if o_base:
            self.openai_base = o_base.rstrip("/")
        o_model = os.getenv("AI_MODEL", "").strip()
        if o_model:
            self.openai_model = o_model
        env_limit = os.getenv("AI_MONTHLY_LIMIT_USD", "").strip()
        if env_limit:
            try:
                self.monthly_limit_usd = max(1.0, float(env_limit))
            except ValueError:
                pass

    @property
    def is_configured(self) -> bool:
        for k in (self.anthropic_key, self.gemini_key, self.openai_key):
            if k and "your_" not in k.lower():
                return True
        return False

    @property
    def active_provider_label(self) -> str:
        if self.anthropic_key and "your_" not in self.anthropic_key.lower():
            return f"Anthropic ({self.anthropic_model})"
        if self.gemini_key and "your_" not in self.gemini_key.lower():
            return f"Gemini ({self.gemini_model})"
        if self.openai_key and "your_" not in self.openai_key.lower():
            return f"OpenAI ({self.openai_model})"
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

    def _record_usage(self, in_tok: int, out_tok: int, in_rate: float, out_rate: float) -> None:
        usage = self._load_usage()
        cost = (in_tok / 1_000_000.0) * in_rate + (out_tok / 1_000_000.0) * out_rate
        usage["input_tokens"] = int(usage.get("input_tokens", 0)) + in_tok
        usage["output_tokens"] = int(usage.get("output_tokens", 0)) + out_tok
        usage["requests"] = int(usage.get("requests", 0)) + 1
        usage["spent_usd"] = round(float(usage.get("spent_usd", 0.0)) + cost, 6)
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
        }

    def _call_llm(
        self,
        user_text: str,
        max_tokens: int = 1600,
        system_prompt: Optional[str] = None,
    ) -> tuple[Optional[str], str]:
        self.reload_env()
        if not self.is_configured:
            return None, "unconfigured"
        usage = self._load_usage()
        if float(usage.get("spent_usd", 0.0)) >= self.monthly_limit_usd:
            logger.warning("AI monthly budget limit $%.2f reached — skipping AI call.", self.monthly_limit_usd)
            return None, "budget_exceeded"
        sys_txt = system_prompt or TRADER_SYSTEM_PROMPT

        # 1. Anthropic Claude
        if self.anthropic_key and "your_" not in self.anthropic_key.lower():
            try:
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
                        "messages": [{"role": "user", "content": user_text}],
                    },
                    timeout=(10, 45),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    u = data.get("usage", {})
                    self._record_usage(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0)), 3.0, 15.0)
                    blocks = data.get("content", [])
                    txt = "\n".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text").strip()
                    if txt:
                        return txt, f"anthropic:{self.anthropic_model}"
                logger.warning("Anthropic error %s: %s", resp.status_code, redact_secrets(resp.text[:200]))
            except Exception as exc:
                logger.warning("Anthropic request failed: %s", redact_secrets(exc))

        # 2. Google Gemini
        if self.gemini_key and "your_" not in self.gemini_key.lower():
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
                resp = requests.post(
                    url,
                    headers={"x-goog-api-key": self.gemini_key, "Content-Type": "application/json"},
                    json={
                        "system_instruction": {"parts": [{"text": sys_txt}]},
                        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
                        "generationConfig": {"temperature": 0.15, "maxOutputTokens": max_tokens},
                    },
                    timeout=(10, 45),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    um = data.get("usageMetadata", {})
                    self._record_usage(int(um.get("promptTokenCount", 0)), int(um.get("candidatesTokenCount", 0)), 0.15, 0.60)
                    cands = data.get("candidates", [])
                    if cands:
                        parts = cands[0].get("content", {}).get("parts", [])
                        txt = "\n".join(str(p.get("text", "")) for p in parts if "text" in p).strip()
                        if txt:
                            return txt, f"gemini:{self.gemini_model}"
                logger.warning("Gemini error %s: %s", resp.status_code, redact_secrets(resp.text[:200]))
            except Exception as exc:
                logger.warning("Gemini request failed: %s", redact_secrets(exc))

        # 3. OpenAI-compatible
        if self.openai_key and "your_" not in self.openai_key.lower():
            try:
                resp = requests.post(
                    f"{self.openai_base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.openai_model,
                        "temperature": 0.15,
                        "max_tokens": max_tokens,
                        "messages": [
                            {"role": "system", "content": sys_txt},
                            {"role": "user", "content": user_text},
                        ],
                    },
                    timeout=(10, 45),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    u = data.get("usage", {})
                    self._record_usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)), 0.5, 1.5)
                    txt = data["choices"][0]["message"]["content"].strip()
                    if txt:
                        return txt, f"openai:{self.openai_model}"
                logger.warning("OpenAI error %s: %s", resp.status_code, redact_secrets(resp.text[:200]))
            except Exception as exc:
                logger.warning("OpenAI request failed: %s", redact_secrets(exc))

        return None, "failed"

    @staticmethod
    def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
        txt = raw_text.strip()
        # Strip markdown fences if present
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
        """Send full candle + TA context to AI Judge and enforce hard Python guardrails."""
        payload = {
            "setup": candidate.context_payload,
            "portfolio": portfolio_state,
        }
        prompt = (
            f"Оцени кандидата {candidate.symbol} ({candidate.direction}) на закрытой свече 15M.\n"
            f"Сработавшие школы ({len(candidate.fired_schools)}): {', '.join(candidate.fired_schools)}.\n"
            f"Верни ТОЛЬКО строгий JSON по заданной схеме.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        raw_txt, provider = self._call_llm(prompt)
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
