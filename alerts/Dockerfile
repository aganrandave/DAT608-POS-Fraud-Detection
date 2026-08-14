FROM python:3.11-slim

WORKDIR /app

COPY alerts/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY alerts/ ./
COPY data/ /app/data/

RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
