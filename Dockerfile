FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU ONLY (massively reduces Docker image size and RAM usage on Render)
RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining dependencies
COPY requirements.txt .
# Remove torch and torchvision from requirements to avoid overriding the CPU version
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Free tier Render uses port 10000 by default, but respects PORT env var.
# We use shell form for CMD to expand the PORT variable.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1
