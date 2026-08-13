# bitnami/spark:3.5.1 was pulled from Docker Hub in Bitnami's 2025 move to
# paid "Bitnami Secure Images" - the tag 404s now. bitnamilegacy/spark is
# the same image content under Bitnami's unmaintained archive org, so this
# keeps the build working with zero other changes, but it's a stopgap:
# bitnamilegacy gets no further updates and could disappear too. Migrating
# to the actively-maintained official apache/spark:3.5.1-python3 image is
# worth doing as a separate, deliberate change (different WORKDIR/user
# layout, needs its own verification).
FROM bitnamilegacy/spark:3.5.1

WORKDIR /app

COPY pipeline/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./
COPY data/ /app/data/

CMD ["spark-submit", "spark_consumer.py"]
