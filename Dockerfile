FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for Python + KenLM build
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install onnxruntime-gpu \
    && pip install https://github.com/kpu/kenlm/archive/master.zip \
    && pip install python-multipart

# Copy app files
COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
