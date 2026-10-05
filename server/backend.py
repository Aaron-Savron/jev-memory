"""Memory decision backend: broad open-weight model, not Open-Jev-2B.

Implements the same prepare/infer contract the scheduler and HTTP layer use:
  identity: str
  prepare(request) -> (records, tokens, longest)
  infer(groups) -> [[{id, score, relation?}]]
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class MemoryBackend:
    def __init__(self):
        from memory_model import MemoryDecisionModel, format_example
        folder = os.environ.get("MEMORY_MODEL_PATH", "/models/memory-v3")
        device = os.environ.get("MEMORY_DEVICE", "cuda:0")
        dtype = os.environ.get("MEMORY_TORCH_DTYPE", "float16")
        self.identity = "memory-v3@qwen3-0.6b-lora"
        self.max_length = 320
        self.batch_size = int(os.environ.get("MEMORY_BATCH_SIZE", "24"))
        if not 1 <= self.batch_size <= 64:
            raise ValueError("MEMORY_BATCH_SIZE must be 1..64")
        path = Path(folder)
        if not (path / "manifest.json").exists():
            raise RuntimeError(f"Memory model not found at {folder}. Train with modal run server/modal_app.py --finetune")
        self.model = MemoryDecisionModel.load(folder, device=device, dtype=dtype)
        manifest = json.loads((path / "manifest.json").read_text())
        self.max_length = int(manifest.get("maxLength", self.model.max_length))
        digest = hashlib.sha256((path / "heads.pt").read_bytes()).hexdigest()[:16]
        self.identity = f"memory-v3@{manifest.get('base', 'qwen3-0.6b')}/{manifest.get('method', 'lora')}/heads-{digest}"
        self._format = format_example
        # Warmup so the first real request does not pay compile/load cost alone.
        self.infer([[{
            "id": "warmup",
            "operation": "retain",
            "context": "Select useful durable memory with source support.",
            "text": "The staging service runs under systemd on the primary host.",
        }]])

    def prepare(self, request):
        """Translate a DecisionRequest into scored items and a token estimate."""
        records = []
        for candidate in request.candidates:
            records.append({
                "id": candidate.id,
                "operation": request.operation,
                "context": request.context,
                "text": candidate.text,
                "previous": getattr(candidate, "previous", None),
            })
        # Token estimate for the scheduler budget: rough 3.2 chars/token on our format.
        total_chars = sum(
            len(self._format(r["operation"], r["context"], r["text"], r["previous"]))
            for r in records
        )
        tokens = max(1, int(total_chars / 3.2))
        longest = min(self.max_length, max(1, tokens // max(1, len(records))))
        return records, tokens, longest

    def infer(self, groups):
        flat = [(gi, ri, record) for gi, group in enumerate(groups) for ri, record in enumerate(group)]
        # Shortest first for tighter batches.
        flat.sort(key=lambda row: len(row[2]["text"]) + len(row[2].get("context", "")))
        output = [[None] * len(group) for group in groups]
        for start in range(0, len(flat), self.batch_size):
            chunk = flat[start : start + self.batch_size]
            items = [
                {
                    "id": record["id"],
                    "operation": record["operation"],
                    "context": record.get("context", ""),
                    "text": record["text"],
                    "previous": record.get("previous"),
                }
                for _, _, record in chunk
            ]
            scored = self.model.score(items)
            for (gi, ri, record), item in zip(chunk, scored):
                if record["operation"] == "relationship":
                    output[gi][ri] = {
                        "id": item["id"],
                        "score": item["score"],
                        "relation": item["relation"],
                    }
                else:
                    output[gi][ri] = {"id": item["id"], "score": item["score"]}
        for group in output:
            for i, decision in enumerate(group):
                if decision is None:
                    raise RuntimeError("Missing model decision")
        return output


# Historical name used by app.py and older scripts.
OpenJevBackend = MemoryBackend
