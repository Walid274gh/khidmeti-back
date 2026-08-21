# ══════════════════════════════════════════════════════════════════════════════
# ai-embed — llama.cpp:server + nomic-embed-text-v1.5 (Render free)
#
# Même image que docker-compose*.yml ; le modèle PUBLIC est téléchargé au boot
# (~90 MB Q8_0) vers le cache conteneur. PORT est imposé par Render → start.sh.
# ══════════════════════════════════════════════════════════════════════════════
FROM ghcr.io/ggml-org/llama.cpp:server

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8012
CMD ["/start.sh"]
