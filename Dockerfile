FROM python:3.11-slim

WORKDIR /app

# install dependency system
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

# expose port
EXPOSE 8000

# start gunicorn
CMD ["gunicorn", "-c", "gunicorn_conf.py", "wsgi:app"]