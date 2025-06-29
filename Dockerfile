FROM python:3.10-slim-bullseye

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    gnupg \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libatspi2.0-0 \
    libwayland-client0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers to user directory
ENV PLAYWRIGHT_BROWSERS_PATH=/app/ms-playwright
RUN python3 -m playwright install chromium --with-deps

# Verify installation
RUN echo "=== Playwright Info ===" && \
    python3 -m playwright --version && \
    echo "=== Installed Browsers ===" && \
    python3 -m playwright install --list && \
    ls -la $PLAYWRIGHT_BROWSERS_PATH/chromium-*/chrome-linux/ && \
    echo "=== Chromium Location ===" && \
    find / -name "chrome-linux" -type d -exec ls -la {} \;

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]