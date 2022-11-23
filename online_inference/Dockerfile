FROM python:3.8.10-slim
# FROM python:3.8.10
# FROM ubuntu:latest


LABEL maintainer="ivan_123_39"

ENV PATH_TO_MODEL="models/log_reg.pkl"

COPY requirements.txt app.py init_env.sh ./
COPY src ./src
COPY models ./models

RUN ./init_env.sh

WORKDIR /online_inference

EXPOSE 9000

CMD [ "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "9000"]