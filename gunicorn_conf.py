import os

bind = os.getenv("BIND", "0.0.0.0:5000")

# 2 vCPU => 2 workers
workers = int(os.getenv("WEB_CONCURRENCY", "2"))

# threads per worker (buat burst request)
threads = int(os.getenv("THREADS", "4"))
worker_class = "gthread"

timeout = int(os.getenv("TIMEOUT", "60"))

# Load model once then fork workers (hemat RAM biasanya)
preload_app = True

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")

# recycle biar stabil jangka panjang
max_requests = int(os.getenv("MAX_REQUESTS", "800"))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", "100"))