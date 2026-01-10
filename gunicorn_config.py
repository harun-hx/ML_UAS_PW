import os

# Bind to the PORT environment variable (Railway/Heroku) or default to 10000 (Render)
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Worker Options
workers = 1
threads = 4
timeout = 120