FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure templates directory exists
RUN mkdir -p templates

COPY . .

# Add healthcheck to verify template exists
RUN if [ ! -f "templates/ubl_2.1_sample.xml" ]; then \
        echo "WARNING: UBL template not found. Ingestion may fail."; \
    fi

CMD ["python", "ingest.py"]