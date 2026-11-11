# Picked up automatically by `gunicorn app:app` (Render).
# Map/related-term queries can take longer than gunicorn's default 30 s on a free instance.
timeout = 180
graceful_timeout = 30
