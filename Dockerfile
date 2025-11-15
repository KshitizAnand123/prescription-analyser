FROM python:3.10-slim

# Install system dependencies: Tesseract, Poppler, FFmpeg
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    ffmpeg \
    && apt-get clean

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python deps from your existing requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
