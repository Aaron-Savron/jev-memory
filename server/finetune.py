"""Large LoRA fine-tune of a small open-weight memory model on T4.

Trains Qwen3-0.6B + task heads on a broad domain-held-out corpus. The frozen
quality fixtures are scored after training and are never part of the train set.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from broad_corpus import corpus, operation_stats, quality_overlap_check
from memory_model import MemoryDecisionModel, format_example
from quality_fixtures import cases as quality_cases


def _to_items(rows: list[dict]) -> list[dict]:
    items = []
    for i, row in enumerate(rows):
        items.append({
            "id": str(i),
            "operation": row["operation"],
            "context": row.get("context", ""),
            "text": row["text"],
            "previous": row.get("previous"),
            "label": row["label"],
        })
    return items


def _labels_for(items: list[dict]) -> list[int]:
    labels = []
    for item in items:
        if item["operation"] == "relationship":
            labels.append({"same": 0, "contradicts": 1, "unrelated": 2}[item["label"]])
        else:
            labels.append(1 if item["label"] else 0)
    return labels


def _evaluate(model: MemoryDecisionModel, items: list[dict]) -> dict:
    batch = 24
    correct = noul_correct = noul_total = 0
    rel_correct = rel_total = 0
    brier = 0.0
    accepted = accepted_ok = 0
    errors = []
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        preds = model.score(chunk)
        labels = _labels_for(chunk)
        for item, pred, label in zip(chunk, preds, labels):
            if item["operation"] == "relationship":
                rel_total += 1
                rel_ok = {"same": 0, "contradicts": 1, "unrelated": 2}[pred["relation"]] == label
                rel_correct += int(rel_ok)
                correct += int(rel_ok)
                if not rel_ok:
                    errors.append({"id": item["id"], "got": pred["relation"], "want": item["label"], "text": item["text"][:80]})
            else:
                p = pred["score"]
                y = float(label)
                ok = (p >= 0.5) == bool(y)
                noul_total += 1
                noul_correct += int(ok)
                correct += int(ok)
                brier += (p - y) ** 2
                if p >= 0.9 or p <= 0.1:
                    accepted += 1
                    accepted_ok += int(ok)
                if not ok:
                    errors.append({"id": item["id"], "got": p, "want": y, "text": item["text"][:80]})
    total = noul_total + rel_total
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "noul": {"correct": noul_correct, "total": noul_total, "accuracy": noul_correct / noul_total if noul_total else 0.0},
        "relationship": {"correct": rel_correct, "total": rel_total, "accuracy": rel_correct / rel_total if rel_total else 0.0},
        "brier": brier / noul_total if noul_total else None,
        "acceptedAt09": accepted,
        "acceptedCorrectAt09": accepted_ok,
        "errors": errors[:12],
    }


def finetune(
    *,
    train_rows: int = 24000,
    calib_rows: int = 2500,
    test_rows: int = 2500,
    epochs: int = 1,
    lr: float = 1.2e-4,
    batch_size: int = 24,
    grad_accum: int = 2,
    max_length: int = 288,
    warmup_ratio: float = 0.03,
    seed: int = 11,
    destination: str = "/models/memory-v3",
):
    import torch
    from torch.utils.data import DataLoader, Dataset

    started = time.perf_counter()
    overlap = quality_overlap_check()
    if overlap["leakCount"]:
        raise RuntimeError(f"Corpus leaks frozen quality fixtures: {overlap['leaked']}")

    splits = {
        "train": corpus("train", seed=seed, max_rows=train_rows),
        "calibration": corpus("calibration", seed=seed, max_rows=calib_rows),
        "test": corpus("test", seed=seed, max_rows=test_rows),
    }
    stats = {name: operation_stats(rows) for name, rows in splits.items()}
    print(json.dumps(stats, indent=2), flush=True)

    model = MemoryDecisionModel.load_base(device="cuda:0", dtype="float16", lora=True)
    model.max_length = max_length
    model.model.train()
    model.heads.train()

    class RowDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    def collate(rows):
        texts = [
            format_example(row["operation"], row.get("context", ""), row["text"], row.get("previous"))
            for row in rows
        ]
        labels = _labels_for([
            {"operation": r["operation"], "label": r["label"], "text": r["text"], "id": str(i)}
            for i, r in enumerate(rows)
        ])
        return texts, labels, [r["operation"] for r in rows]

    dataset = RowDataset(splits["train"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    total_steps = max(1, (len(loader) // grad_accum) * epochs)
    warmup = int(total_steps * warmup_ratio)

    decay, no_decay = [], []
    for name, param in list(model.model.named_parameters()) + list(model.heads.named_parameters()):
        if not param.requires_grad:
            continue
        (no_decay if "bias" in name or "norm" in name.lower() else decay).append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)

    def schedule(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scaler = torch.amp.GradScaler("cuda", enabled=True)
    step = 0
    running = 0.0
    running_n = 0
    running_correct = 0
    running_total = 0
    for epoch in range(epochs):
        for batch_index, (texts, labels, operations) in enumerate(loader):
            model.tokenizer.padding_side = "right"
            encoded = model.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(model.device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                hidden = outputs.hidden_states[-1]
                lengths = encoded["attention_mask"].sum(-1)
                idx = torch.arange(len(texts), device=hidden.device)
                pooled = hidden[idx, lengths - 1].float()
                loss = torch.zeros((), device=model.device)
                for op in ("retain", "relevance", "relationship"):
                    mask = torch.tensor([o == op for o in operations], device=model.device)
                    if not bool(mask.any()):
                        continue
                    logits = model.heads[op](pooled[mask])
                    target = torch.tensor(
                        [label for label, keep in zip(labels, mask.tolist()) if keep],
                        device=model.device,
                        dtype=torch.long,
                    )
                    if op == "relationship":
                        # Relationship is the weakest class; upweight it so
                        # held-out pairs do not collapse to the majority label.
                        loss = loss + 2.0 * torch.nn.functional.cross_entropy(logits, target)
                    else:
                        loss = loss + torch.nn.functional.cross_entropy(logits, target)
                    with torch.no_grad():
                        pred = logits.argmax(-1)
                        running_correct += int((pred == target).sum())
                        running_total += int(target.numel())
            loss = loss / grad_accum
            scaler.scale(loss).backward()
            running += float(loss.detach()) * grad_accum
            running_n += 1
            if (batch_index + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.model.parameters() if p.requires_grad] + list(model.heads.parameters()),
                    1.0,
                )
                lr_now = lr * schedule(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    acc = running_correct / max(1, running_total)
                    print(
                        f"epoch={epoch} step={step}/{total_steps} "
                        f"loss={running / max(1, running_n):.4f} trainAcc={acc:.3f} lr={lr_now:.2e}",
                        flush=True,
                    )
                    running = 0.0
                    running_n = 0
                    running_correct = 0
                    running_total = 0

    model.model.eval()
    model.heads.eval()
    report = {}
    for name, rows in splits.items():
        report[name] = _evaluate(model, _to_items(rows))

    quality_items = []
    for i, case in enumerate(quality_cases()):
        cand = case["request"]["candidates"][0]
        quality_items.append({
            "id": case["name"],
            "operation": case["request"]["operation"],
            "context": case["request"]["context"],
            "text": cand["text"],
            "previous": cand.get("previous"),
            "label": case["expected"],
        })
    quality = _evaluate(model, quality_items)

    manifest = {
        "schema": 3,
        "base": model.base_id(),
        "method": "qwen3-0.6b-lora-multi-task-classification",
        "trainRows": len(splits["train"]),
        "calibrationRows": len(splits["calibration"]),
        "testRows": len(splits["test"]),
        "epochs": epochs,
        "batchSize": batch_size,
        "gradAccum": grad_accum,
        "learningRate": lr,
        "maxLength": max_length,
        "corpus": stats,
        "metrics": report,
        "quality": quality,
        "seconds": time.perf_counter() - started,
        "limitations": "Synthetic broad domain-held-out corpus. Frozen quality fixtures are independent wording families. No real-user or task-success claim.",
    }
    saved = model.save(destination, manifest=manifest)
    print(json.dumps({k: v for k, v in saved.items() if k != "quality"}, indent=2), flush=True)
    print(json.dumps({"quality": {k: quality[k] for k in quality if k != "errors"}}, indent=2), flush=True)
    return {"manifest": manifest, "destination": destination}
