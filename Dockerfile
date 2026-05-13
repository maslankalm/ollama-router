FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_ROUTER_CONFIG_PATH=/etc/ollama-router/config.yaml \
    OLLAMA_ROUTER_HOST=0.0.0.0 \
    OLLAMA_ROUTER_PORT=8080

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

FROM runtime AS test
COPY tests ./tests
RUN pip install --no-cache-dir '.[test]' && pytest -q

FROM runtime AS app
RUN useradd --system --uid 10001 --create-home ollama-router \
    && mkdir -p /etc/ollama-router \
    && chown -R ollama-router:ollama-router /home/ollama-router /etc/ollama-router
USER ollama-router

EXPOSE 8080
CMD ["ollama-router"]
