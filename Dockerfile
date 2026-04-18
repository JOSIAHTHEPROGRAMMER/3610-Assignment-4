# Use slim variant to keep image size down 
FROM python:3.11-slim

# All subsequent commands run from /app inside the container
WORKDIR /app

# Copy requirements first so Docker can cache the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and saved model artifacts
COPY app.py .
COPY models/ ./models/

EXPOSE 8000

# Start the API server, binding to all interfaces so it's reachable outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
