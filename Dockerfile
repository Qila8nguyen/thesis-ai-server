# Use Ubuntu as the base image
FROM --platform=linux/amd64 python:3.10.14-slim-bullseye

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    bash \
    curl \
    clang \
    libfreetype6-dev \
    libpng-dev \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    python3-numpy \
    python3-wheel \
    python3-matplotlib \
    python3-h5py \
    python3-sklearn \
    python3-scipy \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Install Bazelisk
RUN curl -Lo /usr/local/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 && \
    chmod +x /usr/local/bin/bazelisk

# Clone TensorFlow repository
# --depth 1: This option specifies that you want a shallow clone with a history truncated
RUN git clone --branch v2.15.0 https://github.com/tensorflow/tensorflow.git --depth 1 /tensorflow

# Set environment variables to accept default options for TensorFlow build
ENV TF_NEED_CUDA=0
ENV TF_NEED_TENSORRT=0
ENV TF_NEED_OPENCL_SYCL=0

# Build and install TensorFlow
RUN cd /tensorflow && \
    ./configure

# --config opt
RUN cd /tensorflow && \
 bazelisk build //tensorflow/tools/pip_package:build_pip_package && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package  /tmp/tensorflow_pkg 

# Install TensorFlow
RUN pip install /tmp/tensorflow_pkg/*.whl

# Clean up
RUN rm -rf /tensorflow /tmp/tensorflow_pkg

# Copy requirements.txt and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip3 install --no-cache-dir -r /code/requirements.txt

# Copy the rest of the application code
COPY . /code

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run uvicorn server with --reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
