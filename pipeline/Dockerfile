FROM bitnami/spark:3.5.1

WORKDIR /app

COPY pipeline/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./
COPY data/ /app/data/

CMD ["spark-submit", "spark_consumer.py"]
