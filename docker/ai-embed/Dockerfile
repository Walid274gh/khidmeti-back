FROM python:3.11-slim
RUN pip install --no-cache-dir fastapi uvicorn sentence-transformers
WORKDIR /app
COPY server.py .
EXPOSE 8012
CMD ["python", "server.py"]
