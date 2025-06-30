# Use a slim Python image as a base
# We choose a specific version (e.g., 3.9-slim-buster) for reproducibility
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory into the container's /app directory
COPY . .

# Install system dependencies if any are needed for TensorFlow (often not required for slim builds)
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     # Add any system libraries here, e.g., libgfortran5 for some numpy builds
#     && rm -rf /var/lib/apt/lists/*

# Install Python packages
# It's good practice to pin versions for reproducibility, e.g., numpy==1.22.0
# For simplicity, we'll install the latest compatible versions here.
# For TensorFlow, consider using a specific CPU-only or GPU-enabled version.
# This example uses the CPU version.
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    tensorflow

# Command to run the Python script when the container starts
# For Cloud Run, this command is executed as the main process.
# Ensure your script or application starts a web server if it's meant to serve HTTP requests.
# If it's a batch job, it will run and exit.
CMD ["python", "my_netflix-binary.py"]

# Example 'app.py' content you would place in the same directory as the Dockerfile:
#
# import numpy as np
# import pandas as pd
# import tensorflow as tf
#
# print("Python, NumPy, Pandas, and TensorFlow are installed and working!")
#
# # Example usage:
# arr = np.array([1, 2, 3])
# print(f"NumPy array: {arr}")
#
# df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
# print(f"Pandas DataFrame:\n{df}")
#
# hello = tf.constant("Hello, TensorFlow!")
# print(f"TensorFlow constant: {hello.numpy().decode('utf-8')}")
#