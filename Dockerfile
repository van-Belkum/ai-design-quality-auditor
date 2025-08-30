# Simple Dockerfile for the AI Design Quality Auditor
FROM python:3.11-slim

# Install system deps for OCR and PDF rendering (optional but useful)
RUN apt-get update && apt-get install -y \
    tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
