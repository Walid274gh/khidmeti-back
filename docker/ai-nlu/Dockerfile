# docker/ai-nlu/Dockerfile — Khidmeti NLU (dziriBERT 2 têtes, ONNX int8, CPU)
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "onnxruntime>=1.17,<2" \
    "tokenizers>=0.15" \
    "huggingface_hub>=0.23" \
    "numpy>=1.24,<3"

WORKDIR /app
COPY server.py .

EXPOSE 8013

# python:slim n'a pas curl — healthcheck en stdlib
HEALTHCHECK --interval=20s --timeout=5s --retries=6 --start-period=90s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8013/health',timeout=4)"

CMD ["python", "server.py"]
