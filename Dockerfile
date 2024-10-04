FROM python:3.11-slim

WORKDIR /usr/src/main

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
# RUN pip install poetry
# RUN poetry config virtualenvs.create false

# install project dependencies
# COPY pyproject.toml poetry.lock* /usr/src/main/
# RUN poetry install --no-dev --no-interaction --no-ansi

COPY requirements.txt /usr/src/main/
RUN pip install -r requirements.txt


# copy project
COPY . /usr/src/main/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]