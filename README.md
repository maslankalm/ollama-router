# Ollama Router

Minimal Python router that exposes a small OpenAI-compatible API and forwards chat requests to the first healthy Ollama backend by configured priority.

## Implemented endpoints

- `GET /health` - process liveness.
- `GET /status` - configured backends with health state, last check time, and `last_seen` Unix timestamp for the last successful health check.
- `POST /v1/chat/completions` - minimal OpenAI-compatible non-streaming chat endpoint.

Intentionally not implemented yet: embeddings, model listing, streaming, tool calls, retries across multiple backends inside a single chat request, auth on the router itself.

## Config

YAML config is loaded from `OLLAMA_ROUTER_CONFIG_PATH` or `/etc/ollama-router/config.yaml` by default.

```yaml
healthcheck_interval_seconds: 30
healthcheck_timeout_seconds: 5
request_timeout_seconds: 120

backends:
  - name: primary-local
    priority: 10
    endpoint: http://ollama-primary.example.internal:11434

  - name: secondary-local
    priority: 20
    endpoint: http://ollama-secondary.example.internal:11434

  - name: cloud-fallback
    priority: 100
    endpoint: https://ollama.com
    api_key_env: OLLAMA_CLOUD_API_KEY
```

Backend fields:

- `name`: unique backend id.
- `priority`: lower number wins.
- `endpoint`: base Ollama URL, without a trailing slash.
- `api_key`: optional bearer token.
- `api_key_env`: optional environment variable name for a bearer token. If set, this wins over `api_key` when the env var exists.

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
OLLAMA_ROUTER_CONFIG_PATH=examples/config.yaml ollama-router
```

Test a chat call:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"say ok"}]}'
```

## GitHub Actions / GHCR

The repository includes `.github/workflows/docker.yml`. On pushes to `main` or tags it:

1. builds the Docker `test` stage, which runs the pytest suite;
2. publishes the `app` stage to `ghcr.io/<owner>/<repo>` for `linux/amd64` and `linux/arm64`.

PRs build the test stage but do not push images. Image builds are independent of runtime configuration. Local runtime YAML files under `config/*.yaml` are ignored by both Git and Docker build context.

## Kubernetes + Tailscale

`ollama-router` can run in Kubernetes with a `tailscale/tailscale` sidecar so the router Pod can reach private Ollama backends without exposing those backends publicly.

This repo stays deployment-agnostic: concrete backend endpoints, namespace names, auth values, and cluster manifests belong in the environment repo. The usual shape is:

- `ollama-router` container serves the OpenAI-compatible API.
- `tailscale/tailscale` sidecar joins the private network from the same Pod.
- Router config points backends at endpoints reachable from that Pod.
- Cloud fallback credentials and Tailscale auth are supplied through Kubernetes Secrets.

## Docker

```bash
docker build --target test -t ollama-router:test .
docker build --target app -t ollama-router:local .
```

For Kubernetes, mount the YAML config to `/etc/ollama-router/config.yaml` and pass cloud backend credentials as environment variables.
