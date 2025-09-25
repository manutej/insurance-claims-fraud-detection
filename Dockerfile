# Multi-stage Dockerfile for Insurance Claims Fraud Detection System
# Optimized for production with security hardening

# Build stage
FROM python:3.10-slim-bullseye AS builder

# Build arguments
ARG VERSION=latest
ARG COMMIT_SHA=unknown
ARG BUILD_DATE=unknown

# Set build-time labels
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.revision=$COMMIT_SHA
LABEL org.opencontainers.image.title="Insurance Claims Fraud Detection"
LABEL org.opencontainers.image.description="Production-ready fraud detection system for insurance claims"
LABEL org.opencontainers.image.vendor="Insurance Claims Analysis Team"

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

# Install system dependencies required for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and switch to app user for build
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml setup.py ./

# Create virtual environment and install dependencies
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip setuptools wheel && \
    /app/venv/bin/pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY main.py fraud_detection_demo.py example_usage.py ./

# Install the application
RUN /app/venv/bin/pip install -e .

# Production stage
FROM python:3.10-slim-bullseye AS production

# Production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PORT=8000 \
    WORKERS=4 \
    MAX_WORKERS=8 \
    TIMEOUT=30 \
    KEEP_ALIVE=2 \
    LOG_LEVEL=info

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Security updates
    ca-certificates \
    # Monitoring tools
    curl \
    # Health check utilities
    netcat-openbsd \
    # Process management
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/models /app/tmp && \
    chown -R appuser:appgroup /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appgroup /app/venv /app/venv

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --from=builder --chown=appuser:appgroup /app/src ./src/
COPY --from=builder --chown=appuser:appgroup /app/main.py /app/fraud_detection_demo.py /app/example_usage.py ./

# Copy configuration files
COPY --chown=appuser:appgroup pytest.ini tox.ini pyproject.toml ./
COPY --chown=appuser:appgroup data/ ./data/

# Create startup script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Function to handle graceful shutdown
graceful_shutdown() {
    echo "Received shutdown signal, stopping application..."
    kill -TERM "$child" 2>/dev/null || true
    wait "$child"
    exit 0
}

# Set up signal handlers
trap graceful_shutdown SIGTERM SIGINT

# Health check function
health_check() {
    python -c "
import sys
import importlib.util
try:
    spec = importlib.util.spec_from_file_location('main', '/app/main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print('Application loaded successfully')
    sys.exit(0)
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
"
}

# Validate environment
echo "Starting Insurance Claims Fraud Detection System..."
echo "Python version: $(python --version)"
echo "Environment: ${ENVIRONMENT:-development}"
echo "Log level: ${LOG_LEVEL}"

# Run health check
if ! health_check; then
    echo "Application failed health check, exiting..."
    exit 1
fi

# Start application based on mode
if [ "${MODE:-api}" = "api" ]; then
    echo "Starting API server on port ${PORT}..."
    exec gunicorn main:app \
        --bind "0.0.0.0:${PORT}" \
        --workers "${WORKERS}" \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --timeout "${TIMEOUT}" \
        --keep-alive "${KEEP_ALIVE}" \
        --worker-class uvicorn.workers.UvicornWorker \
        --access-logfile - \
        --error-logfile - \
        --log-level "${LOG_LEVEL}" &
elif [ "${MODE}" = "worker" ]; then
    echo "Starting background worker..."
    exec python -m src.worker &
elif [ "${MODE}" = "scheduler" ]; then
    echo "Starting scheduler..."
    exec python -m src.scheduler &
else
    echo "Starting in CLI mode..."
    exec python main.py "$@" &
fi

child=$!
wait "$child"
EOF

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["/app/entrypoint.sh"]

# Add metadata labels
LABEL org.opencontainers.image.source="https://github.com/insurance-claims/fraud-detection"
LABEL org.opencontainers.image.documentation="https://insurance-claims-pipeline.readthedocs.io/"
LABEL org.opencontainers.image.licenses="MIT"