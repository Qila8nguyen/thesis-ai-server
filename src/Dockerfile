FROM ubuntu:22.04

MAINTAINER Khoan <duckhoan.ds@gmail.com>

# Setup environments
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg locales git apt-utils && \
    apt-get -y install --no-install-recommends python3.10 python3-pip python3-dev build-essential && \
    apt -y install make cmake gcc g++ && \
    apt -y install libsndfile1-dev && \  
    pip install --upgrade pip && \
    locale-gen en_US.UTF-8  && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update

COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

RUN pip install nvidia-pyindex && pip install nvidia-cuda-nvcc

# 
WORKDIR /code

COPY . /code

# Make port 80 available to the world outside this container
EXPOSE 8000

# Run uvicorn server with --reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]