FROM python:3.11-slim

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY models.py tasks.py rewards.py environment.py main.py openenv.yaml ./
COPY server/ ./server/

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
