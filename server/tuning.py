"""Multi-task memory head on frozen Open-Jev backbone/LoRA features.

What changed vs the first tuner
-------------------------------
* Corpus is domain-held-out (see train_corpus.py), not name-template held-out.
* Trains the shared scalar head on retain/relevance (BCE) *and* relationship
  option ranking (softmax CE over the three choice prompts).
* Optional 2-layer MLP probe when a linear probe underfits.
* Calibration selects temperature and model variant; test domains stay frozen.
* The frozen quality fixture set is scored after training and never trained on.
"""
import hashlib
import io
import json
import time
from pathlib import Path


def _module(kind: str, dim: int, hidden: int):
    import torch
    if kind == "linear":
        head = torch.nn.Linear(dim, 1, device="cuda", dtype=torch.float32)
    elif kind == "mlp":
        head = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden, device="cuda", dtype=torch.float32),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, 1, device="cuda", dtype=torch.float32),
        )
    else:
        raise ValueError(kind)
    return head


def _init_from_linear(head, original_state):
    import torch
    if isinstance(head, torch.nn.Linear):
        head.load_state_dict(original_state)
    else:
        # Seed the first projection from the released linear head so early
        # steps do not destroy the pretrained direction.
        with torch.no_grad():
            head[0].weight.copy_(original_state["weight"])
            head[0].bias.copy_(original_state["bias"])


def _encode(backend, rows):
    """Return per-example hidden features (one row per model prompt)."""
    import torch
    from protocol import DecisionRequest
    model = backend.predictor.scorer.model
    batches: list = []
    handle = model.head.register_forward_pre_hook(
        lambda module, inputs: batches.append(inputs[0].detach().float().cpu())
    )
    try:
        records = []
        prompt_counts = []
        for i, row in enumerate(rows):
            if row["operation"] == "relationship":
                request = DecisionRequest(operation="relationship", context=row["context"],
                                          candidates=[{"id": str(i), "text": row["text"], "previous": row["previous"]}],
                                          deadlineMs=10000)
            else:
                request = DecisionRequest(operation=row["operation"], context=row.get("context", "Select durable memory."),
                                          candidates=[{"id": str(i), "text": row["text"]}], deadlineMs=10000)
            prepared = backend.prepare(request)[0]
            # relationship compiles to one choice record with 3 option prompts;
            # noul compiles to one record with one prompt.
            prompt_counts.append(sum(len(backend.prompts(record)) for record in prepared))
            records.extend(prepared)
        backend._score(records)
    finally:
        handle.remove()
    flat = torch.cat(batches, dim=0)
    expected = sum(prompt_counts)
    if flat.shape[0] != expected:
        raise RuntimeError(f"Feature/prompt mismatch: {flat.shape[0]} vs {expected}")
    features = []
    cursor = 0
    for count in prompt_counts:
        features.append(flat[cursor:cursor + count])
        cursor += count
    return features, prompt_counts


def _noul_prob(head, group, temperature):
    import torch
    # Noul renders one prompt; head emits a scalar logit; serving stacks [0, logit].
    logit = head(group).reshape(-1)[0]
    return torch.sigmoid(logit / temperature)


def _choice_probs(head, group, temperature):
    import torch
    # Choice renders one prompt per option; each gets a scalar logit.
    logits = head(group).reshape(-1) / temperature
    return torch.softmax(logits, dim=-1)


def tune(backend, kind: str = "mlp", hidden: int = 256, steps: int = 2500):
    import torch
    from train_corpus import corpus, quality_overlap_check
    from quality_fixtures import cases as quality_cases
    started = time.perf_counter()
    overlap = quality_overlap_check()
    if overlap["leakCount"]:
        raise RuntimeError(f"Training corpus leaks frozen quality fixtures: {overlap['leaked']}")

    splits = {name: corpus(name) for name in ("train", "calibration", "test")}
    print({name: len(rows) for name, rows in splits.items()}, flush=True)

    original_batch = backend.batch_size
    backend.batch_size = 16
    model = backend.predictor.scorer.model
    original_linear = {k: v.detach().clone().float().cpu() for k, v in model.head.state_dict().items()}
    dim = original_linear["weight"].shape[1]

    features = {}
    for name, rows in splits.items():
        features[name], _ = _encode(backend, rows)
        print(f"Encoded {name}: {len(rows)} examples", flush=True)

    def batch_loss(head, rows, feats, temperature, rng):
        noul_idx = [i for i, r in enumerate(rows) if r["operation"] != "relationship"]
        rel_idx = [i for i, r in enumerate(rows) if r["operation"] == "relationship"]
        loss = torch.zeros((), device="cuda")
        counts = 0
        if noul_idx:
            logits = []
            labels = []
            for i in noul_idx:
                group = feats[i].cuda()
                logits.append(head(group).reshape(-1)[0])
                labels.append(float(rows[i]["label"]))
            logits = torch.stack(logits)
            labels = torch.tensor(labels, device="cuda")
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels)
            counts += len(noul_idx)
        if rel_idx:
            # Rank the gold relation option above the other two.
            gold = {"same": 0, "contradicts": 1, "unrelated": 2}
            # answer_keys order from protocol.compile_items: same, contradicts, unrelated
            losses = []
            for i in rel_idx:
                group = feats[i].cuda()
                logits = head(group).reshape(-1) / temperature
                target = torch.tensor([gold[rows[i]["label"]]], device="cuda")
                losses.append(torch.nn.functional.cross_entropy(logits.unsqueeze(0), target))
            loss = loss + torch.stack(losses).mean()
            counts += len(rel_idx)
        return loss, counts

    def eval_split(head, rows, feats, temperature):
        head.eval()
        noul_correct = noul_total = 0
        rel_correct = rel_total = 0
        brier_sum = 0.0
        accepted = accepted_ok = 0
        with torch.no_grad():
            for i, row in enumerate(rows):
                group = feats[i].cuda()
                if row["operation"] == "relationship":
                    probs = _choice_probs(head, group, temperature).cpu()
                    pred = int(probs.argmax())
                    gold = {"same": 0, "contradicts": 1, "unrelated": 2}[row["label"]]
                    rel_total += 1
                    rel_correct += int(pred == gold)
                else:
                    p = float(_noul_prob(head, group, temperature).cpu())
                    y = float(bool(row["label"]))
                    noul_total += 1
                    noul_correct += int((p >= 0.5) == bool(y))
                    brier_sum += (p - y) ** 2
                    if p >= 0.9 or p <= 0.1:
                        accepted += 1
                        accepted_ok += int((p >= 0.5) == bool(y))
        head.train()
        total = noul_total + rel_total
        correct = noul_correct + rel_correct
        return {
            "correct": correct, "total": total,
            "accuracy": correct / total if total else 0.0,
            "noul": {"correct": noul_correct, "total": noul_total,
                     "accuracy": noul_correct / noul_total if noul_total else 0.0},
            "relationship": {"correct": rel_correct, "total": rel_total,
                             "accuracy": rel_correct / rel_total if rel_total else 0.0},
            "brier": brier_sum / noul_total if noul_total else None,
            "acceptedAt09": accepted, "acceptedCorrectAt09": accepted_ok,
        }

    best = None
    variants = [(kind, hidden)] if kind != "both" else [("linear", 0), ("mlp", hidden)]
    for vkind, vhidden in variants:
        for rate in (0.0003, 0.001, 0.003):
            head = _module(vkind, dim, vhidden or 256)
            _init_from_linear(head, original_linear)
            optimizer = torch.optim.AdamW(head.parameters(), lr=rate, weight_decay=0.01)
            rows = splits["train"]
            feats = features["train"]
            rng = list(range(len(rows)))
            import random as _random
            noul_idx = [i for i in rng if rows[i]["operation"] != "relationship"]
            rel_idx = [i for i in rng if rows[i]["operation"] == "relationship"]
            for step in range(steps):
                pick_n = min(64, len(noul_idx))
                pick_r = min(32, len(rel_idx))
                sample = _random.sample(noul_idx, pick_n) + (_random.sample(rel_idx, pick_r) if pick_r else [])
                sample_rows = [rows[i] for i in sample]
                sample_feats = [feats[i] for i in sample]
                loss, _ = batch_loss(head, sample_rows, sample_feats, 1.0, rng)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                if step % 250 == 0 or step == steps - 1:
                    print(f"{vkind} lr={rate} step={step} loss={float(loss):.4f}", flush=True)
            with torch.no_grad():
                val_logits = []
                for row, feat in zip(splits["calibration"], features["calibration"]):
                    group = feat.cuda()
                    if row["operation"] == "relationship":
                        val_logits.append(None)
                    else:
                        val_logits.append(float(head(group).reshape(-1)[0].cpu()))
                best_temp = 1.0
                best_nll = None
                for temperature in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
                    nll = 0.0
                    n = 0
                    for row, logit in zip(splits["calibration"], val_logits):
                        if logit is None:
                            continue
                        p = 1 / (1 + pow(2.718281828, -logit / temperature))
                        y = float(bool(row["label"]))
                        nll -= y * __import__("math").log(max(p, 1e-9)) + (1 - y) * __import__("math").log(max(1 - p, 1e-9))
                        n += 1
                    nll /= max(n, 1)
                    if best_nll is None or nll < best_nll:
                        best_nll, best_temp = nll, temperature
            report = eval_split(head, splits["calibration"], features["calibration"], best_temp)
            if best is None or report["accuracy"] > best["accuracy"]:
                state = {k: v.detach().clone().float().cpu() for k, v in head.state_dict().items()}
                best = {"accuracy": report["accuracy"], "temperature": best_temp, "rate": rate,
                        "kind": vkind, "hidden": vhidden or 0, "state": state, "calibration": report}
                print("new best", vkind, rate, report, flush=True)

    head = _module(best["kind"], dim, best["hidden"] or 256)
    head.load_state_dict(best["state"])
    report = {}
    for name, rows in splits.items():
        report[name] = eval_split(head, rows, features[name], best["temperature"])

    # Frozen quality fixture set: never trained on. Score it the same way serving does.
    quality_rows = []
    for case in quality_cases():
        op = case["request"]["operation"]
        cand = case["request"]["candidates"][0]
        quality_rows.append({
            "domain": "quality", "operation": op, "context": case["request"]["context"],
            "text": cand["text"], "previous": cand.get("previous"),
            "label": case["expected"], "name": case["name"],
        })
    quality_feats, _ = _encode(backend, quality_rows)
    quality_metrics = eval_split(head, quality_rows, quality_feats, best["temperature"])
    # Per-case answers for the quality set, matching evaluate() conventions.
    quality_answers = []
    with torch.no_grad():
        for row, feat in zip(quality_rows, quality_feats):
            group = feat.cuda()
            if row["operation"] == "relationship":
                probs = _choice_probs(head, group, best["temperature"]).cpu()
                pred = ["same", "contradicts", "unrelated"][int(probs.argmax())]
                score = float(probs.max())
            else:
                p = float(_noul_prob(head, group, best["temperature"]).cpu())
                pred = p >= 0.5
                score = p if pred else 1 - p
            quality_answers.append({"name": row["name"], "answer": pred, "expected": row["label"],
                                    "correct": pred == row["label"], "confidence": score, "operation": row["operation"]})

    buffer = io.BytesIO()
    payload = {
        "schema": 2,
        "kind": best["kind"],
        "hidden": best["hidden"],
        "state": {k: v for k, v in best["state"].items()},
    }
    torch.save(payload, buffer)
    artifact = buffer.getvalue()
    manifest = {
        "schema": 2,
        "parentModel": backend.identity,
        "headSha256": hashlib.sha256(artifact).hexdigest(),
        "headKind": best["kind"],
        "headHidden": best["hidden"],
        "temperature": best["temperature"],
        "dtype": "float16",
        "training": "frozen backbone and LoRA; multi-task probe (retain/relevance BCE + relationship CE); domain-held-out corpus",
        "learningRate": best["rate"],
        "steps": steps,
        "corpus": {name: len(rows) for name, rows in splits.items()},
        "metrics": report,
        "quality": quality_metrics,
        "qualityAnswers": quality_answers,
        "seconds": time.perf_counter() - started,
        "limitations": "Synthetic domain-held-out corpus. Frozen quality fixtures are independent wording families, not real users. No agent task-success or production-calibration claim.",
    }
    backend.batch_size = original_batch
    return {"head": artifact, "manifest": manifest}
