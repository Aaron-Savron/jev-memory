FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY *.py ./
RUN useradd --create-home --uid 10001 jev && mkdir /models && chown jev:jev /models
USER jev
ENV HF_HOME=/models HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
EXPOSE 8094
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8094", "--workers", "1", "--no-access-log"]
