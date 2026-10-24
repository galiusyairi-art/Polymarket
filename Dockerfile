# Fly.io (or any container host): the same program Render runs.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY polybot ./polybot
ENV PYTHONUNBUFFERED=1 POLYBOT_DB=/data/polybot.sqlite PORT=8080
EXPOSE 8080
CMD ["python", "-m", "polybot", "--host", "0.0.0.0", "--no-browser"]
