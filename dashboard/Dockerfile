FROM python:3.11-slim

WORKDIR /app

COPY dashboard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ ./
COPY data/ /app/data/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
