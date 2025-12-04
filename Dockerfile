FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

RUN pip install --no-cache-dir -e .

EXPOSE 5000

CMD ["python", "scripts/inference_server.py"]