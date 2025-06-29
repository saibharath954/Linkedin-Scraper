FROM python:3.10-slim

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

# Install Node.js (required for Playwright)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

WORKDIR /app
COPY . .

# Install Playwright via pip (Python package)
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers to a known location
ENV PLAYWRIGHT_BROWSERS_PATH=/app/ms-playwright
RUN playwright install chromium
RUN playwright install-deps

# Verify browser installation
RUN echo "=== Listing browser files ===" && \
    find / -name "chrome-linux" -type d -exec ls -la {} \; && \
    echo "=== Playwright version ===" && \
    playwright --version && \
    echo "=== Installed browsers ===" && \
    playwright install --list && \
    echo "=== Checking chromium path ===" && \
    ls -la /app/ms-playwright/chromium-*/chrome-linux/
    
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]