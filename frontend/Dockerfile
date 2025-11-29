FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# -w enables auto-reload for development
# --port 8501 sets the port
# --host 0.0.0.0 is required for Docker networking
CMD ["chainlit", "run", "app.py", "-w", "--port", "8501", "--host", "0.0.0.0"]