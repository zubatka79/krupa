FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN useradd --system --uid 10001 --home-dir /app krupa \
    && mkdir -p data/ai_trader_v2/cache logs config \
    && chown -R krupa:krupa data logs config

USER krupa

CMD ["python", "-u", "ai_trader_v2/run_ai_trader.py", "--demo"]
