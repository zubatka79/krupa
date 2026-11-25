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
    && mkdir -p data logs \
    && chown -R krupa:krupa data logs

USER krupa

CMD ["python", "run_bot.py", "--config", "config_ada.json"]
