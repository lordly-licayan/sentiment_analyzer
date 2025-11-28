# --------------------------
# Stage 1: Build stage
# --------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only essential build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install into a separate directory
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# --------------------------
# Stage 2: Runtime stage
# --------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy app code
COPY --from=builder /app /app

# Expose port Cloud Run expects
EXPOSE 8080

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
