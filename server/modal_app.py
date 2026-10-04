"""Bounded T4 evaluation and optional authenticated server deployment."""
import json
import os
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("svrn-jev-memory")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git")
         .pip_install_from_requirements(ROOT / "requirements.lock")
         .env({"HF_HOME": "/models/hf", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1", "JEV_TORCH_DTYPE": "float16"})
         .add_local_dir(ROOT, remote_path="/srv/jev-memory", ignore=[".venv", "__pycache__", "results-*.json"]))
cache = modal.Volume.from_name("svrn-jev-memory-models", create_if_missing=True)


@app.function(image=image, gpu="T4", cpu=2, memory=12288, timeout=600,
              max_containers=1, retries=0, volumes={"/models": cache})
def adapt():
    import sys
    sys.path.insert(0, "/srv/jev-memory")
    from backend import OpenJevBackend
    from tuning import tune
    result = tune(OpenJevBackend())
    destination = Path('/models/memory-head')
    destination.mkdir(exist_ok=True)
    (destination/'head.pt').write_bytes(result['head'])
    (destination/'manifest.json').write_text(json.dumps(result['manifest'], indent=2)+'\n')
    cache.commit()
    return result


@app.function(image=image, gpu="T4", cpu=2, memory=12288, timeout=600,
              max_containers=1, retries=0, volumes={"/models": cache})
def evaluate(use_head: bool = False):
    import sys
    import time
    import torch
    sys.path.insert(0, "/srv/jev-memory")
    from backend import OpenJevBackend
    from protocol import DecisionRequest
    from fixtures import cases
    started = time.perf_counter()
    if use_head:
        os.environ['JEV_MEMORY_HEAD'] = '/models/memory-head'
    backend = OpenJevBackend()
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
        rows.append({**case, "answer": answer, "decision": decision, "correct": answer == case["expected"],
                     "latencyMs": elapsed, "inputTokens": tokens, "longestSequence": longest})
        print(f"{case['name']}: {answer} expected={case['expected']} {elapsed:.0f}ms", flush=True)
    from fastapi.testclient import TestClient
    from app import create_app
    os.environ['JEV_MEMORY_TOKEN'] = 'synthetic-test-token-only-0123456789'
    probe = DecisionRequest(operation='relevance',context='Deploy the staging app', candidates=[{'id':str(i),'text':'Staging runs under systemd on the Atlas host.'} for i in range(12)],deadlineMs=3000)
    start = time.perf_counter()
    with TestClient(create_app(lambda: backend)) as client:
        response = client.post('/v1/memory/decide',json=probe.model_dump(),headers={'authorization':'Bearer '+os.environ['JEV_MEMORY_TOKEN']})
    serving = {'status':response.status_code,'candidates':12,'wallMs':1000*(time.perf_counter()-start)}
    if response.status_code == 200:
        serving['timing'] = response.json()['timing']
        if len(response.json()['decisions']) != 12:
            raise RuntimeError('Incomplete serving response')
    latencies = sorted(row["latencyMs"] for row in rows)
    return {"model": backend.identity, "provenance": backend.predictor.provenance,
            "gpu": torch.cuda.get_device_name(), "loadSeconds": loaded,
            "peakAllocatedGiB": torch.cuda.max_memory_allocated() / 2**30,
            "peakReservedGiB": torch.cuda.max_memory_reserved() / 2**30,
            "correct": sum(r["correct"] for r in rows), "total": len(rows),
            "p50Ms": latencies[len(rows)//2], "p95Ms": latencies[int(len(rows)*0.95)-1],
            "servingSmoke":serving,"rows": rows, "limitations": "Synthetic smoke set and one 12-candidate serving request, no serving-load or task-success claim."}


@app.local_entrypoint()
def main(tune_head: bool = False, use_head: bool = False):
    if tune_head:
        result = adapt.remote()
        destination = ROOT / 'memory-head'
        destination.mkdir(exist_ok=True)
        (destination/'head.pt').write_bytes(result['head'])
        (destination/'manifest.json').write_text(json.dumps(result['manifest'], indent=2)+'\n')
        print(json.dumps(result['manifest'], indent=2))
        return
    result = evaluate.remote(use_head)
    destination = ROOT / ("results-t4-tuned.json" if use_head else "results-t4.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print(f"Saved {destination}")
