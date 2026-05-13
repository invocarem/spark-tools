# sglang_docker

SGLang runtime operations via Docker containers — mirrors [sglang_runtime](../sglang_runtime/) but runs inside Docker instead of a local Python venv.

## Run

```bash
python3 sglang_docker/sglang_docker.py [--verbose] <subcommand>
```

## Subcommands

| Command | Description |
|---------|-------------|
| `pull` | Pull the Docker image locally or via SSH to remote hosts |
| `launch` | Start sglang in a Docker container (`docker run -d --gpus all --network host`) |
| `stop` | `docker stop` + `docker rm` the container(s) |
| `logs` | `docker logs --tail N` on the container |
| `scan` | Probe a running server's `/health`, `/v1/models`, `/get_server_info` |
| `benchmark` | Run API benchmark via OpenAI-compatible endpoint |
| `measure` | Capture GPU/CPU/memory snapshots |

## Configuration

Same preset resolution as `sglang_runtime` — CLI flags > env vars > preset JSON > defaults.

Presets file defaults to `sglang_docker/model_presets.json`. Each preset uses an `image` field instead of `venv_path`:

```json
{
  "my-model": {
    "image": "scitrera/dgx-spark-sglang:0.5.11",
    "model_path": "~/huggingface/my-model",
    "tp": 2,
    "port": 30000,
    "sglang_args": ["--trust-remote-code", "--enable-metrics"]
  }
}
```

## Container naming

- Solo mode: `sglang-{preset_name}` (or `sglang-server` if no preset)
- Cluster mode: `sglang-{preset_name}-node{idx}`

## Example

```bash
# Pull the image on two hosts
python3 sglang_docker/sglang_docker.py pull --preset my-model --hosts spark-01 spark-02

# Launch solo on a remote host
python3 sglang_docker/sglang_docker.py launch --preset my-model --host spark-01

# Launch cluster
python3 sglang_docker/sglang_docker.py launch --mode cluster --preset my-model --hosts spark-01,spark-02

# Check logs
python3 sglang_docker/sglang_docker.py logs --preset my-model --host spark-01

# Stop
python3 sglang_docker/sglang_docker.py stop --preset my-model --host spark-01
```
