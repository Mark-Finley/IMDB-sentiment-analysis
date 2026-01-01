# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 8000 (Render will map this)
EXPOSE 8000

# Run the FastAPI app with uvicorn
# Render sets the PORT environment variable, default to 8000 if not set
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
