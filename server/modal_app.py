"""Bounded T4 fine-tune and evaluation for the broad memory model."""
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("svrn-memory-v3")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements(ROOT / "requirements.lock")
    .env({
        "HF_HOME": "/models/hf",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "MEMORY_TORCH_DTYPE": "float16",
        "MEMORY_MODEL_PATH": "/models/memory-v3",
    })
    .add_local_dir(ROOT, remote_path="/srv/memory", ignore=[".venv", "__pycache__", "results-*.json", "memory-head"])
)
cache = modal.Volume.from_name("svrn-memory-v3", create_if_missing=True)


@app.function(image=image, gpu="T4", cpu=4, memory=16384, timeout=10800,
              max_containers=1, retries=0, volumes={"/models": cache})
def finetune(train_rows: int = 24000, epochs: int = 1, lr: float = 1.2e-4, batch_size: int = 24):
    import sys
    sys.path.insert(0, "/srv/memory")
    from finetune import finetune as run
    result = run(
        train_rows=train_rows,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        destination="/models/memory-v3",
    )
    cache.commit()
    return result["manifest"]


@app.function(image=image, gpu="T4", cpu=4, memory=16384, timeout=900,
              max_containers=1, retries=0, volumes={"/models": cache})
def evaluate(suite: str = "quality"):
    import sys
    import time
    import torch
    sys.path.insert(0, "/srv/memory")
    from backend import MemoryBackend
    from protocol import DecisionRequest
    if suite == "quality":
        from quality_fixtures import cases
    else:
        from fixtures import cases
    started = time.perf_counter()
    backend = MemoryBackend()
    cache.commit()
    loaded = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for case in cases():
        body = DecisionRequest(**case["request"])
        records, tokens, longest = backend.prepare(body)
        start = time.perf_counter()
        decision = backend.infer([records])[0][0]
        elapsed = 1000 * (time.perf_counter() - start)
        answer = decision.get("relation", decision["score"] >= 0.5)
        operation = case["request"]["operation"]
        rows.append({
            **case,
            "operation": operation,
            "answer": answer,
            "decision": decision,
            "correct": answer == case["expected"],
            "latencyMs": elapsed,
            "inputTokens": tokens,
            "longestSequence": longest,
        })
        print(f"{case['name']}: {answer} expected={case['expected']} {elapsed:.0f}ms", flush=True)
    from fastapi.testclient import TestClient
    from app import create_app
    os.environ["JEV_MEMORY_TOKEN"] = "synthetic-test-token-only-0123456789"
    probe = DecisionRequest(
        operation="relevance",
        context="Deploy the staging app",
        candidates=[{"id": str(i), "text": "Staging runs under systemd on the Atlas host."} for i in range(12)],
        deadlineMs=3000,
    )
    start = time.perf_counter()
    with TestClient(create_app(lambda: backend)) as client:
        response = client.post(
            "/v1/memory/decide",
            json=probe.model_dump(),
            headers={"authorization": "Bearer " + os.environ["JEV_MEMORY_TOKEN"]},
        )
    serving = {"status": response.status_code, "candidates": 12, "wallMs": 1000 * (time.perf_counter() - start)}
    if response.status_code == 200:
        serving["timing"] = response.json()["timing"]
        if len(response.json()["decisions"]) != 12:
            raise RuntimeError("Incomplete serving response")
    latencies = sorted(row["latencyMs"] for row in rows)
    by_operation = {}
    for row in rows:
        bucket = by_operation.setdefault(row["operation"], {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(row["correct"])
    for bucket in by_operation.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
    return {
        "model": backend.identity,
        "gpu": torch.cuda.get_device_name(),
        "loadSeconds": loaded,
        "peakAllocatedGiB": torch.cuda.max_memory_allocated() / 2**30,
        "peakReservedGiB": torch.cuda.max_memory_reserved() / 2**30,
        "correct": sum(r["correct"] for r in rows),
        "total": len(rows),
        "byOperation": by_operation,
        "p50Ms": latencies[len(latencies) // 2],
        "p95Ms": latencies[int(len(latencies) * 0.95) - 1] if len(latencies) > 20 else latencies[-1],
        "servingSmoke": serving,
        "rows": rows,
        "limitations": "Synthetic held-out fixtures. Not production calibration or agent task success.",
    }


@app.local_entrypoint()
def main(finetune_model: bool = False, suite: str = "quality", train_rows: int = 24000, epochs: int = 1):
    if finetune_model:
        manifest = finetune.remote(train_rows, epochs)
        destination = ROOT / "memory-v3-manifest.json"
        destination.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({k: v for k, v in manifest.items() if k not in ("metrics", "quality")}, indent=2))
        print("quality", json.dumps({k: manifest.get("quality", {}).get(k) for k in ("correct", "total", "accuracy", "noul", "relationship")}, indent=2))
        print(f"Saved {destination}")
        return
    result = evaluate.remote(suite)
    suffix = f"-{suite}" if suite != "smoke" else ""
    destination = ROOT / f"results-memory-v3{suffix}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print(f"Saved {destination}")
