"""Benchmark — OpenAI-compatible chat completion latency/throughput."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from statistics import mean


def run_benchmark(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    requests: int,
    timeout_sec: int,
) -> dict[str, float | int] | None:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    latencies = []
    failures = 0
    for i in range(requests):
        start = time.perf_counter()
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            failures += 1
            print(f"[benchmark] request {i + 1} failed: {exc}", file=sys.stderr)
            continue
        latencies.append(time.perf_counter() - start)

    if not latencies:
        print("No successful benchmark requests.", file=sys.stderr)
        return None

    total_time = sum(latencies)
    rps = len(latencies) / total_time if total_time > 0 else 0.0
    sorted_lat = sorted(latencies)

    def pct(p: float) -> float:
        idx = min(len(sorted_lat) - 1, int((p / 100.0) * len(sorted_lat)))
        return sorted_lat[idx]

    return {
        "successful_requests": len(latencies),
        "failed_requests": failures,
        "avg_latency_sec": round(mean(latencies), 4),
        "p50_latency_sec": round(pct(50), 4),
        "p95_latency_sec": round(pct(95), 4),
        "throughput_rps": round(rps, 3),
    }
