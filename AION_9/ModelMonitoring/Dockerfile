FROM python:3.8-slim-buster
LABEL "usecase"="AION_8"
LABEL "usecase_test"="AION_8_test"
COPY  requirements.txt requirements.txt
COPY  input_drift.py input_drift.py
COPY  config.json config.json
COPY  code.py code.py

RUN pip install --no-cache-dir -r requirements.txt