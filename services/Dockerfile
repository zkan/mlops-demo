# Use official Python image as base
FROM python:3.13.2-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code and model to the container
COPY app.py .
COPY artifacts/model.pkl ./artifacts/

# Expose the FastAPI port
EXPOSE 8000

# Command to run FastAPI server with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
