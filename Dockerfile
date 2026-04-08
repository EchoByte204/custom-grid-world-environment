# ============================================================
# Dockerfile
# Builds the Warehouse Robot Navigation Environment container.
#
# Build:  docker build -t warehouse-nav-env .
# Run:    docker run -p 7860:7860 --env-file .env warehouse-nav-env
# ============================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="openenv-participant"
LABEL description="Warehouse Robot Navigation Environment for OpenEnv competition"

# Set working directory
WORKDIR /workspace

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer cache optimization)
COPY requirements.txt .

# Install Python dependencies
# --only-binary=:all: avoids needing C compiler
RUN pip install --no-cache-dir --only-binary=:all: \
    numpy \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose HF Spaces port
EXPOSE 7860

# Health check — competition validator uses this
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
