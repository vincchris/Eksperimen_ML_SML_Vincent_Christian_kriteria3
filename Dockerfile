FROM python:3.12.7-slim

WORKDIR /app

COPY MLProject/ .

RUN pip install -r requirements.txt

CMD ["python", "modelling.py"]