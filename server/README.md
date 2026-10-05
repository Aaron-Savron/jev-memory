# Decision server

Authenticated, bounded scoring for the TypeScript engine. The server has no memory database. It returns supplied IDs and probabilities, and cannot apply edits or grant permissions.

The model is a **Qwen3-0.6B** backbone with LoRA and three classification heads (`retain`, `relevance`, `relationship`). It replaces the earlier Open-Jev-2B probe. Training data is a broad synthetic corpus (engineering, product, ops, personal, research, scientific, and other domains) with domain-held-out and template-family-held-out splits.

## Docker

```sh
docker build -t jev-memory-server server
docker run --gpus all --rm -p 127.0.0.1:8094:8094 \
  -e JEV_MEMORY_TOKEN -e MEMORY_MODEL_PATH=/models/memory-v3 \
  -v jev-models:/models jev-memory-server
```

Set `JEV_MEMORY_TOKEN` to a random token of at least 24 characters before running. Put a TLS proxy in front of remote deployments. Run one worker per model replica; multiple Uvicorn workers duplicate weights.

`MEMORY_MODEL_PATH` points at a folder with `manifest.json`, `heads.pt`, and optionally `adapter/`. `MEMORY_DEVICE` defaults to `cuda:0`. `MEMORY_BATCH_SIZE` defaults to 24.

## Endpoints

`GET /health` checks the process. Authenticated `/ready` checks warmup and scheduler readiness. `/capabilities` returns the exact model identity and limits. Pin its `model` value in the client as `JEV_MEMORY_MODEL`.

`POST /v1/memory/decide` accepts `operation`, `context`, `candidates`, and `deadlineMs`. Operations: `retain`, `relevance`, and `relationship`. A relationship candidate also includes `previous`.

The queue has 64 slots and an eight-request quota per authentication token. Relevance requests take priority over background writes. Microbatching waits at most 10 ms and caps batch tokens. Queued/expired work is skipped; executing CUDA work can finish after a client disconnect. Late results are discarded.

Default logs do not include excerpts. Use metadata-only observability and disable access logs in remote deployments.

## Training

```sh
modal run server/modal_app.py --finetune-model --train-rows 24000 --epochs 1
modal run server/modal_app.py --suite quality
```

Fine-tuning writes `manifest.json`, `heads.pt`, and the LoRA adapter to the Modal volume under `MEMORY_MODEL_PATH` (default `/models/memory-v3`). It never reads CoFound's database.

Generalization checks (not production calibration):

* Train / calibration / test splits use **disjoint topic domains** and **disjoint template families**.
* The frozen `quality_fixtures.py` set is scored after training and is never in the train corpus.
* Loss on contrastive near-miss pairs stays positive (it does not collapse to 0.000). If train accuracy hits 1.0 while test accuracy drops, the corpus is too templated.

## Native Open-Jev API (optional)

The package can still call an existing native Open-Jev deployment or the hosted API directly. Set `OPENJEV_API_KEY`, optionally set `OPENJEV_URL`, and construct `OpenJevApiProvider`. It sends the memory decision as Noul or Choice questions to `/v1/systemone`, so no package server is required.

```ts
new OpenJevApiProvider({
  url: process.env.OPENJEV_URL ?? 'https://api.openjev.sh',
  apiKey: process.env.OPENJEV_API_KEY,
  model: process.env.OPENJEV_MODEL ?? 'openjev',
});
```

Use `HttpDecisionProvider` when you want the bounded `/v1/memory/decide` contract and the package server's scheduler and batching.

## Modal deployment

For a persistent deployment, create a Modal secret named `jev-memory-server` containing `JEV_MEMORY_TOKEN`, then run:

```sh
modal deploy server/modal_serve.py
```

That recipe scales to zero when idle. No persistent endpoint is deployed by the experiment commands.
