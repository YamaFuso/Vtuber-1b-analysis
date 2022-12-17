# syntax=docker/dockerfile:1
FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y default-jdk scala wget vim software-properties-common python3-pip curl unzip libpq-dev build-essential libssl-dev libffi-dev python3-dev&& \
    apt-get clean

RUN wget https://archive.apache.org/dist/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3-scala2.13.tgz && \
    tar xvf spark-3.3.1-bin-hadoop3-scala2.13.tgz && \
    mv spark-3.3.1-bin-hadoop3-scala2.13/ /usr/local/spark && \
    ln -s /usr/local/spark spark

WORKDIR /app
COPY . .

RUN pip3 install cython numpy && pip3 install pytest pyspark pandas
ENV PYSPARK_PYTHON=python3

EXPOSE 3000
