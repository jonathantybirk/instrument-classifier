# Base image
FROM python:3.13-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt

RUN pip install -r requirements.txt --no-cache-dir --verbose

RUN dvc pull data/processed

COPY src/instrument_classifier/train.py src/instrument_classifier/train.py
COPY src/instrument_classifier/data.py src/instrument_classifier/data.py
COPY src/instrument_classifier/model.py src/instrument_classifier/model.py
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/instrument_classifier/train.py"]
