# Base image
FROM python:3.13-bookworm

# System dependencies
RUN apt-get update && apt-get install -y dumb-init && update-ca-certificates

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port for FastAPI
EXPOSE 8192

# Run FastAPI server
CMD ["dumb-init", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8192"]
