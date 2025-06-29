# Use the official Playwright image with Python
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

# Set work directory
WORKDIR /app

# Install system dependencies (already included in base image)
# Just ensure essential tools are present
RUN apt-get update && \
    apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install browsers to known location
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN mkdir -p /ms-playwright && \
    playwright install chromium && \
    playwright install --list

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]