# Use official PyTorch image for GPU/CPU support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY main.py .

# Command to run the script
CMD ["python", "main.py"]