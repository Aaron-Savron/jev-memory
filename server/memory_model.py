"""Broad memory decision model on a small open-weight backbone.

Replaces Open-Jev-2B. A Qwen3-0.6B backbone (LoRA-tunable) plus task-specific
classification heads scores:

* retain      -> P(durable memory)
* relevance   -> P(useful for the request)
* relationship -> P(same | contradicts | unrelated)

The public decision contract is unchanged: score in [0,1], relationship as a
label string.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

TASKS = ("retain", "relevance", "relationship")
RELATIONSHIP_LABELS = ("same", "contradicts", "unrelated")


def format_example(operation: str, context: str, text: str, previous: str | None = None) -> str:
    """Compact classifier input. Keep this identical at train and serve time."""
    if operation == "relationship":
        return (
            f"<memory:relationship>\n"
            f"context: {context}\n"
            f"previous: {previous}\n"
            f"candidate: {text}\n"
        )
    return (
        f"<memory:{operation}>\n"
        f"context: {context}\n"
        f"candidate: {text}\n"
    )


class MemoryDecisionModel:
    """LoRA backbone + three classification heads."""

    def __init__(self, model, tokenizer, heads, lora=None, device="cuda:0", max_length: int = 320):
        self.model = model
        self.tokenizer = tokenizer
        self.heads = heads
        self.lora = lora
        self.device = device
        self.max_length = max_length

    @staticmethod
    def base_id() -> str:
        return "Qwen/Qwen3-0.6B"

    @staticmethod
    def load_base(device: str = "cuda:0", dtype="float16", lora: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = MemoryDecisionModel.base_id()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Last-token pooling assumes right padding. Qwen defaults to left.
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype if isinstance(dtype, torch.dtype) else getattr(torch, str(dtype), torch.float16),
            trust_remote_code=True,
        )
        model.to(device)
        hidden = model.config.hidden_size
        heads = torch.nn.ModuleDict({
            "retain": torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 2, device=device)),
            "relevance": torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 2, device=device)),
            "relationship": torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 3, device=device)),
        })
        adapter = None
        if lora:
            from peft import LoraConfig, get_peft_model
            adapter = LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, adapter)
            model.print_trainable_parameters()
        return MemoryDecisionModel(model, tokenizer, heads, lora=adapter, device=device)

    def save(self, destination: str | Path, manifest: dict | None = None) -> dict:
        import torch
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        heads_path = destination / "heads.pt"
        torch.save({k: v.detach().cpu() for k, v in self.heads.state_dict().items()}, heads_path)
        payload: dict = {
            "schema": 3,
            "base": self.base_id(),
            "headSha256": hashlib.sha256(heads_path.read_bytes()).hexdigest(),
            "maxLength": self.max_length,
        }
        if self.lora is not None and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(destination / "adapter")
            payload["adapter"] = "adapter"
        if manifest:
            payload.update(manifest)
        (destination / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    @classmethod
    def load(cls, folder: str | Path, device: str = "cuda:0", dtype="float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        folder = Path(folder)
        manifest = json.loads((folder / "manifest.json").read_text())
        model_id = manifest.get("base", cls.base_id())
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype if isinstance(dtype, torch.dtype) else getattr(torch, str(dtype), torch.float16),
            trust_remote_code=True,
        )
        if manifest.get("adapter"):
            model = PeftModel.from_pretrained(model, folder / manifest["adapter"])
        model.to(device)
        hidden = model.config.hidden_size
        heads = torch.nn.ModuleDict({
            "retain": torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 2, device=device)),
            "relevance": torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 2, device=device)),
            "relationship": torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 3, device=device)),
        })
        heads.load_state_dict(torch.load(folder / "heads.pt", map_location=device, weights_only=True))
        heads.to(device)
        return cls(model, tokenizer, heads, device=device, max_length=int(manifest.get("maxLength", 320)))

    def _hidden(self, texts: list[str]):
        import torch
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        lengths = encoded["attention_mask"].sum(-1)
        idx = torch.arange(len(texts), device=hidden.device)
        # Last non-pad token for each sequence.
        return hidden[idx, lengths.to(hidden.device) - 1].float()

    def score(self, items: list[dict]) -> list[dict]:
        """items: [{operation, context, text, previous?}]. Returns list of dicts."""
        if not items:
            return []
        import torch
        texts = [
            format_example(
                item["operation"],
                item.get("context", ""),
                item["text"],
                item.get("previous"),
            )
            for item in items
        ]
        hidden = self._hidden(texts)
        out: list[dict] = []
        with torch.no_grad():
            for i, item in enumerate(items):
                h = hidden[i : i + 1]
                op = item["operation"]
                logits = self.heads[op](h)[0]
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
                if op == "relationship":
                    index = max(range(3), key=probs.__getitem__)
                    out.append({
                        "id": item["id"],
                        "score": float(probs[index]),
                        "relation": RELATIONSHIP_LABELS[index],
                        "probs": probs,
                    })
                else:
                    # probs[0]=false, probs[1]=true
                    out.append({"id": item["id"], "score": float(probs[1]), "probs": probs})
        return out


def encode_texts(model: MemoryDecisionModel, texts: list[str]):
    return model._hidden(texts)
