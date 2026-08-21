# docker/ai-vision/Dockerfile — Khidmeti vision (SigLIP2 zero-shot, ONNX int8, CPU)
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "onnxruntime>=1.17,<2" \
    "pillow>=10" \
    "huggingface_hub>=0.23" \
    "numpy>=1.24,<3"

WORKDIR /app
COPY server.py .

EXPOSE 8015

# python:slim n'a pas curl — healthcheck en stdlib.
# start-period long : premier boot = téléchargement ~470 MB + chargement.
HEALTHCHECK --interval=20s --timeout=5s --retries=6 --start-period=300s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8015/health',timeout=4)"

CMD ["python", "server.py"]
