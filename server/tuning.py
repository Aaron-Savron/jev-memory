"""Adapt only the scalar decision head, keeping the released backbone/LoRA frozen."""
import hashlib
import io
import json
import math
import time


def tune(backend):
    import torch
    from protocol import DecisionRequest
    from training_data import examples
    model = backend.predictor.scorer.model
    original = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
    splits = {name: examples(name) for name in ("train", "calibration", "test")}
    original_batch = backend.batch_size
    backend.batch_size = 16
    features = {}
    baseline = {}
    started = time.perf_counter()
    for name, rows in splits.items():
        vectors = []
        handle = model.head.register_forward_pre_hook(lambda module, inputs: vectors.append(inputs[0].detach().clone()))
        try:
            records = []
            for i, row in enumerate(rows):
                request = DecisionRequest(operation=row["operation"], context=row["context"], candidates=[{"id": str(i), "text": row["text"]}], deadlineMs=10000)
                records.extend(backend.prepare(request)[0])
            probabilities = backend._score(records)
        finally:
            handle.remove()
        features[name] = torch.cat(vectors)
        baseline[name] = [p[1] for p in probabilities]
        print(f"Encoded {name}: {len(rows)} examples", flush=True)
    labels = {name: torch.tensor([row["label"] for row in rows], device="cuda", dtype=torch.float32) for name, rows in splits.items()}
    train_x = features["train"]
    best = None
    # Calibration chooses hyperparameters/temperature; the test split is untouched.
    for rate in (0.0001, 0.0003, 0.001):
        head = torch.nn.Linear(train_x.shape[1], 1, device="cuda", dtype=torch.float32)
        head.load_state_dict(original)
        optimizer = torch.optim.Adam(head.parameters(), lr=rate)
        for step in range(600):
            logits = head(train_x).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels["train"])
            loss = loss + 0.01 * (head.weight-original["weight"]).square().sum()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            val_logits = head(features["calibration"]).squeeze(-1)
            for temperature in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
                nll = torch.nn.functional.binary_cross_entropy_with_logits(val_logits / temperature, labels["calibration"]).item()
                if best is None or nll < best["nll"]:
                    best = {"nll": nll, "temperature": temperature, "rate": rate, "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
    def metrics(values, rows):
        truth = [bool(r["label"]) for r in rows]
        correct = sum((p >= 0.5) == y for p, y in zip(values, truth))
        accepted = [(p, y) for p, y in zip(values, truth) if p >= 0.9]
        return {"correct": correct, "total": len(rows), "acceptedAt09": len(accepted),
                "acceptedCorrectAt09": sum(y for _, y in accepted),
                "brier": sum((p-float(y))**2 for p,y in zip(values,truth))/len(rows)}
    head = torch.nn.Linear(train_x.shape[1], 1, device="cuda", dtype=torch.float32)
    head.load_state_dict(best["state"])
    report = {}
    with torch.no_grad():
        for name, rows in splits.items():
            probabilities = torch.sigmoid(head(features[name]).squeeze(-1) / best["temperature"]).cpu().tolist()
            report[name] = {"baseline": metrics(baseline[name], rows), "tuned": metrics(probabilities, rows)}
    buffer = io.BytesIO()
    torch.save({k:v.detach().cpu() for k,v in best["state"].items()}, buffer)
    artifact = buffer.getvalue()
    manifest = {"schema": 1, "parentModel": backend.identity, "headSha256": hashlib.sha256(artifact).hexdigest(),
                "temperature": best["temperature"], "dtype": "float16", "training": "frozen backbone and LoRA; scalar-head BCE with L2; 600 steps; calibration-only hyperparameter selection",
                "learningRate": best["rate"], "metrics": report,
                "seconds": time.perf_counter()-started,
                "limitations": "Synthetic template-family split. Test names and wording held out; task families overlap. Does not establish real-user precision, task success, or injection resistance."}
    backend.batch_size = original_batch
    return {"head": artifact, "manifest": manifest}
