# Use a slim Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script and dataset
COPY scripts/ ./scripts/
COPY Crop_recommendation.csv ./

# Produce artifacts by running the training script
# This will create the /app/model_artifacts directory inside the image
RUN python scripts/train_and_export.py

# Copy the FastAPI application code
COPY app/ ./app/

# Expose the port the app will run on
EXPOSE 8000
ENV PORT=8000

# Start the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]