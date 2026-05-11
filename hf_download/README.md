# Hugging Face model download

Download a full Hugging Face repo (models, tokenizer, processor configs, etc.) with [`huggingface_hub.snapshot_download`](https://huggingface.co/docs/huggingface_hub/guides/download#download-an-entire-repository) only—no `transformers` dependency in this script.

## Setup

```bash
pip install huggingface_hub
```

For private or gated repos, set a token (see [HF authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)):

```bash
export HF_TOKEN=hf_...
```

## Usage

```bash
python hf_download/download.py --model-id <org/repo>
```

Optional output parent directory (default is `/data/hf`):

```bash
python hf_download/download.py --model-id meta-llama/Llama-3.2-1B --save-dir ./models
```

Files are written under `<save-dir>/<org_repo>` (slashes in the repo id become underscores, e.g. `meta-llama_Llama-3.2-1B`).

The downloader skips `*.h5`, `*.ot`, and `*.msgpack` to avoid redundant large artifacts when they are not needed. While the run is in progress, a periodic line is printed with free disk space on the destination volume.

## Exit codes

- `0` — download finished and the local path was printed.
- `1` — empty model id or download failed.
