# Fly.io (or any container host): the same program Render runs.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY polybot ./polybot
# (MALLOC_ARENA_MAX: the C library keeps a pool of memory for every few threads, 27 of them for the bot's 58 threads, and
# each keeps what was freed in it; two pools keep far less. 20.11.2026, the bot ended after eight hours out of memory)
ENV PYTHONUNBUFFERED=1 POLYBOT_DB=/data/polybot.sqlite PORT=8080 MALLOC_ARENA_MAX=2
EXPOSE 8080
CMD ["python", "-m", "polybot", "--host", "0.0.0.0", "--no-browser"]
