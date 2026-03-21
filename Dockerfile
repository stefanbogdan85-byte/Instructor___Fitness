FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels -r requirements-prod.txt

FROM python:3.12-slim

WORKDIR /app

ENV TFHUB_CACHE_DIR=/opt/tfhub_cache
ENV FITNESS_DATA_DIR=/app/data
RUN mkdir -p /opt/tfhub_cache /app/data

COPY --from=builder /wheels /wheels
COPY requirements-prod.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/wheels -r requirements-prod.txt \
    && rm -rf /wheels
COPY gradio_app ./gradio_app
COPY src ./src

RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')"

EXPOSE 80

VOLUME ["/app/models", "/app/data"]

CMD ["python", "gradio_app/app.py"]
