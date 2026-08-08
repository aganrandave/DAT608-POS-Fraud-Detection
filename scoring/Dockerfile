FROM python:3.13-slim

WORKDIR /app

COPY scoring/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY scoring/ ./
COPY models/mlflow_registry.py /app/../models/mlflow_registry.py
COPY data/ /app/data/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
