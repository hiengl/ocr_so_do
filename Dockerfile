# Use the official PyTorch image with PyTorch 2.0.1, CUDA 11.8, and cuDNN 8 pre-installed
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .
# COPY pytorch.txt .

# Install Python dependencies
# PyTorch and torchvision are already included in the base image
# RUN pip install --no-cache-dir -r pytorch.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]