# Decision server

Authenticated, bounded scoring for the TypeScript engine. The server has no memory database. It returns supplied IDs and probabilities, and cannot apply edits or grant permissions.

## Docker

```sh
docker build -t jev-memory-server server
docker run --gpus all --rm -p 127.0.0.1:8094:8094 \
  -e JEV_MEMORY_TOKEN -v jev-models:/models jev-memory-server
```

Set `JEV_MEMORY_TOKEN` to a random token of at least 24 characters before running. Put a TLS proxy in front of remote deployments. Run one worker per model replica; multiple Uvicorn workers duplicate weights.

The loader pins Open-Jev source, the 2B adapter, the base model, and Python dependencies. Defaults: FP16 reference backend, no prefix cache, batch size 16, and 512 tokens per scoring sequence including the chat template. Longer evidence is rejected, never truncated.

`JEV_DEVICE=cpu` enables CPU inference. Use `JEV_TORCH_DTYPE=bfloat16` for the reference CPU recipe; numerical changes require evaluation. The laptop does not need to load these weights.

## Endpoints

`GET /health` checks the process. Authenticated `/ready` checks warmup and scheduler readiness. `/capabilities` returns the exact model identity and limits. Pin its `model` value in the client as `JEV_MEMORY_MODEL`.

`POST /v1/memory/decide` accepts `operation`, `context`, `candidates`, and `deadlineMs`. Operations: `retain`, `relevance`, and `relationship`. A relationship candidate also includes `previous`.

The queue has 64 slots and an eight-request quota per authentication token. Relevance requests take priority over background writes. Microbatching waits at most 10 ms and caps batch tokens. Queued/expired work is skipped; executing CUDA work can finish after a client disconnect. Late results are discarded.

Default logs do not include excerpts. Use metadata-only observability and disable access logs in remote deployments.

## Modal T4

```sh
modal run server/modal_app.py
modal run server/modal_app.py --tune-head
modal run server/modal_app.py --use-head
```

These are bounded experiments with one T4 and synthetic fixtures. The tuning command keeps the backbone and LoRA frozen, trains the scalar head, and selects temperature on a separate calibration split. It writes `server/memory-head` locally and to the model volume. It never reads CoFound's database.

The included experimental head improved the synthetic evaluations, but is not a real-user calibration certificate. It is opt-in through `JEV_MEMORY_HEAD=/path/to/memory-head`. The loader checks its parent identity, hash, dtype, and temperature. Relationship decisions keep the original head because that task was not trained.

For a persistent deployment, create a Modal secret named `jev-memory-server` containing `JEV_MEMORY_TOKEN`, then run:

```sh
modal deploy server/modal_serve.py
```

That recipe scales to zero when idle. Enable the optional head by adding `JEV_MEMORY_HEAD=/models/memory-head` to the secret after running the tuning command. No persistent endpoint is deployed by the experiment commands.
