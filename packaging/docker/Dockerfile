# Dockerfile for agybridge containerized deployment
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy package source
COPY . /app

# Install agybridge with MCP support
RUN pip install --no-cache-dir .[mcp]

# Expose HTTP port
EXPOSE 8791

# Default environment
ENV AGYBRIDGE_TOKEN=""
ENV PYTHONUNBUFFERED=1

# Run HTTP server binding to 0.0.0.0 inside container
CMD ["agybridge", "serve", "--host", "0.0.0.0", "--port", "8791"]

