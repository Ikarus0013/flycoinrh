FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    DATA_DIR=/data

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY relay.py safety.py ./

# Runs as root on purpose: Railway volumes mount /data root-owned.
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn relay:app --host 0.0.0.0 --port ${PORT} --timeout-graceful-shutdown 5 --no-server-header --no-access-log"]
