# docker/ai-stt/Dockerfile — Khidmeti STT (CPU) : faster-whisper int8 (défaut)
# ou wav2vec2-CTC ONNX (STT_ENGINE=ctc). PyAV, embarqué par faster-whisper,
# fournit ffmpeg et décode l'audio pour LES DEUX moteurs → aucun paquet système.
FROM python:3.11-slim

# onnxruntime arrive déjà comme dépendance transitive (VAD silero) — épinglé
# explicitement : c'est le moteur d'inférence du mode ctc, pas un détail de VAD.
RUN pip install --no-cache-dir \
    "faster-whisper>=1.0,<2" \
    "onnxruntime>=1.17,<2"

WORKDIR /app
# ctc_decode.py : décodeur faisceau+LM (portage du kernel LM) — obligatoire
# avec STT_ENGINE=ctc, importé par server.py au module level.
COPY server.py ctc_decode.py .

EXPOSE 8014

# python:slim n'a pas curl — healthcheck en stdlib.
# start-period long : premier boot = téléchargement des poids (small 484 MB, medium 1.5 GB) + chargement.
HEALTHCHECK --interval=20s --timeout=5s --retries=6 --start-period=600s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8014/health',timeout=4)"

CMD ["python", "server.py"]
