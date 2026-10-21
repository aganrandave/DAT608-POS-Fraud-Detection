# bitnami/spark:3.5.1 was pulled from Docker Hub in Bitnami's 2025 move to
# paid "Bitnami Secure Images" - the tag 404s now. bitnamilegacy/spark is
# the same image content under Bitnami's unmaintained archive org, so this
# keeps the build working with zero other changes, but it's a stopgap:
# bitnamilegacy gets no further updates and could disappear too. Migrating
# to the actively-maintained official apache/spark:3.5.1-python3 image is
# worth doing as a separate, deliberate change (different WORKDIR/user
# layout, needs its own verification).
FROM bitnamilegacy/spark:4.0.0

WORKDIR /app

COPY pipeline/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./
COPY data/ /app/data/

# --packages must be a spark-submit launch flag, not something set from
# spark_consumer.py's SparkSession.builder.config() - package resolution
# happens during spark-submit's own launch phase, before the JVM classloader
# is fixed and before any Python driver code runs, so setting it from inside
# the driver is silently too late (confirmed: it did nothing, no download
# ever ran, same "Failed to find data source: kafka" error either way).
CMD ["spark-submit", "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1", "spark_consumer.py"]
