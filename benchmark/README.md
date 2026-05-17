# Benchmark scripts

Small CLI utilities for measuring **throughput / latency** and **task quality** against OpenAI-compatible inference servers (SGLang on port 30000, vLLM on port 8000). No package install step — run with `python3` and set `PYTHONPATH` to this directory.

```bash
cd benchmark
export PYTHONPATH=.
```

## Scripts

| Script | Purpose | Requires |
|--------|---------|----------|
| `benchmark_sglang.py` | Wrapper around `python -m sglang.bench_serving` | SGLang installed in the active Python env |
| `benchmark_vllm.py` | Wrapper around `python -m vllm bench serve` | vLLM 0.6+ in the active Python env |
| `task_benchmark.py` | JSONL task suite → chat completions → regex/contains checkers | stdlib only |
| `benchmark_qwen3_397b.py` | Preset env + delegates to `benchmark_sglang.py` for Qwen3.5-397B GPTQ | Same as SGLang wrapper |

`benchmark_common.py` holds shared helpers (env parsing, `GET /v1/models`, JSON flags).

### Serving benchmarks (throughput / latency)

Both serving wrappers use the same **served vs HF model** split:

- **Served id** — the `model` field sent to the API (`--model` on the wrapper, or `BENCHMARK_SERVED_MODEL`). Auto-detected from `GET /v1/models` when omitted.
- **HF id** — Hugging Face repo for tokenizer / synthetic prompts (`--hf-model`, `BENCHMARK_HF_MODEL`, or `--tokenizer`).

Defaults use a small **random** dataset (3 prompts, 128 input / 128 output tokens) so smoke tests finish quickly.

**SGLang** (default `http://127.0.0.1:30000`, backend `sglang-oai-chat`):

```bash
python3 benchmark_sglang.py

python3 benchmark_sglang.py \
  --base-url http://127.0.0.1:30000 \
  --model qwen3.5-35b \
  --hf-model Qwen/Qwen3.5-35B-A17B \
  --num-prompts 10
```

**vLLM** (default `http://127.0.0.1:8000`, backend `openai-chat`):

```bash
python3 benchmark_vllm.py

python3 benchmark_vllm.py \
  --base-url http://127.0.0.1:8000 \
  --model my-served-name \
  --hf-model Qwen/Qwen3.5-35B \
  --num-prompts 10
```

Run inside the same environment as the server (venv, or `docker exec` into the vLLM / SGLang container).

### Task benchmark (pass rate)

`task_benchmark.py` sends curated prompts from a JSONL file and scores replies with simple checkers (`regex`, `contains`, `contains_all`). Use this when you care about **correctness on fixed tasks**, not tokens/sec.

```bash
python3 task_benchmark.py --input task_benchmark_seed.jsonl

python3 task_benchmark.py \
  --input task_benchmark_seed.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model my-served-name
```

See the module docstring in `task_benchmark.py` for the JSONL schema.

### Qwen3.5-397B preset

`benchmark_qwen3_397b.py` applies repo-specific `BENCHMARK_*` defaults (10 prompts, 512→256 tokens, served `qwen3.5-397b`) then runs `benchmark_sglang.py`. Use `--preset none` to skip presets; extra flags are forwarded.

```bash
python3 benchmark_qwen3_397b.py
python3 benchmark_qwen3_397b.py --preset none --num-prompts 5
```

## Environment variables

### Serving (`benchmark_sglang.py` / `benchmark_vllm.py`)

| Variable | Role |
|----------|------|
| `BENCHMARK_BASE_URL` | Server base URL (overrides runtime-specific defaults below) |
| `SGLANG_BASE_URL` | SGLang default when `BENCHMARK_BASE_URL` unset |
| `VLLM_BASE_URL` | vLLM default when `BENCHMARK_BASE_URL` unset |
| `BENCHMARK_BACKEND` | Bench backend name |
| `BENCHMARK_DATASET` | Dataset name (default `random`) |
| `BENCHMARK_NUM_PROMPTS` | Number of prompts |
| `BENCHMARK_RANDOM_INPUT_LEN` / `BENCHMARK_RANDOM_OUTPUT_LEN` | Token lengths for random datasets |
| `BENCHMARK_MAX_CONCURRENCY` | Optional concurrency cap |
| `BENCHMARK_SERVED_MODEL` / `BENCHMARK_MODEL` | Served model id for the API |
| `BENCHMARK_HF_MODEL` | HF repo for bench `--model` |
| `BENCHMARK_TOKENIZER` | HF tokenizer path or repo |
| `BENCHMARK_EXTRA_REQUEST_BODY` | JSON object merged into the request body (SGLang: `--extra-request-body`; vLLM: `--extra-body`) |
| `BENCHMARK_PRESERVE_SEPARATE_REASONING` | SGLang only: do not set `separate_reasoning: false` |
| `BENCHMARK_PRESERVE_THINKING` | Do not inject `chat_template_kwargs.enable_thinking: false` |

CLI flags override env vars. Unknown arguments are passed through to the underlying bench tool.

### Task (`task_benchmark.py`)

| Variable | Default |
|----------|---------|
| `TASK_BENCH_BASE_URL` | `http://127.0.0.1:8000` |
| `TASK_BENCH_MODEL` | auto from `/v1/models` |
| `TASK_BENCH_TEMPERATURE` | `0` |
| `TASK_BENCH_MAX_TOKENS` | `512` |
| `TASK_BENCH_TIMEOUT_SEC` | `120` |
| `TASK_BENCH_PRESERVE_SEPARATE_REASONING` | (SGLang-style servers) |
| `TASK_BENCH_PRESERVE_THINKING` | (Qwen3) |

## Stack UI

The Stack UI **Serving** benchmark tab calls `benchmark/benchmark_sglang.py` via `POST /api/benchmark/serving` (venv runtime can use the preset’s `venv_path` for Python). Task benchmarks use `task_benchmark.py` via `POST /api/benchmark/task`.

## Sample result files

`sglang-oai-chat_*.jsonl` in this directory are example bench outputs, not inputs.
