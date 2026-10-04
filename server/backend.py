"""Pinned Open-Jev loader. Never silently truncate an evidence span."""
import math
import os
import json
from importlib.metadata import distribution
from pathlib import Path

SOURCE_REVISION = "bd4118882f733574a3250a4b65fe4d884130c08b"
CHECKPOINT_REVISION = "0c7aa498b1627be8da4acf34c863ff0ee0a92785"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


class OpenJevBackend:
    def __init__(self):
        from huggingface_hub import snapshot_download
        from jev.serving import load_predictor
        from jev.api import compile_request, candidate_prompts
        from protocol import compile_items
        if os.environ.get("JEV_LOAD_4BIT") or os.environ.get("JEV_LOAD_8BIT") or os.environ.get("JEV_DEVICE_MAP"):
            raise RuntimeError("This serving recipe requires the unquantized reference backend")
        dtype = os.environ.setdefault("JEV_TORCH_DTYPE", "float16")
        device = os.environ.get("JEV_DEVICE", "cuda:0")
        self.identity = f"open-jev-2b@{CHECKPOINT_REVISION}/{SOURCE_REVISION}/reference-{dtype}"
        self.batch_size = int(os.environ.get("JEV_BATCH_SIZE", "16"))
        if not 1 <= self.batch_size <= 32:
            raise ValueError("JEV_BATCH_SIZE must be 1..32")
        path = snapshot_download("ZefanCai/Open-Jev-2B", revision=CHECKPOINT_REVISION,
                                 allow_patterns=["package/checkpoint/*"])
        snapshot_download("Qwen/Qwen3.5-2B", revision=BASE_REVISION,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        self.predictor = load_predictor(checkpoint=Path(path) / "package/checkpoint",
                                       device=device, max_length=512,
                                       batch_size=self.batch_size, prefix_cache=False)
        actual = self.predictor.provenance
        installed = json.loads(distribution("open-jev").read_text("direct_url.json") or "{}")
        code_revision = actual.get("code_commit") or installed.get("vcs_info", {}).get("commit_id")
        if actual["base_revision"] != BASE_REVISION or code_revision != SOURCE_REVISION:
            raise RuntimeError("Loaded source or base revision does not match the pinned recipe")
        self.original_head = None
        self.memory_head = None
        self.memory_temperature = None
        custom = os.environ.get("JEV_MEMORY_HEAD")
        if custom:
            import torch
            import hashlib
            folder = Path(custom)
            manifest = json.loads((folder / 'manifest.json').read_text())
            data = (folder / 'head.pt').read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if manifest['parentModel'] != self.identity or manifest['headSha256'] != digest or manifest['dtype'] != dtype:
                raise RuntimeError('Memory head manifest does not match the parent model, bytes, or dtype')
            temperature = float(manifest['temperature'])
            if not math.isfinite(temperature) or temperature <= 0:
                raise RuntimeError('Invalid memory head temperature')
            model = self.predictor.scorer.model
            self.original_head = {k:v.detach().clone() for k,v in model.head.state_dict().items()}
            self.memory_head = torch.load(folder/'head.pt', map_location=device, weights_only=True)
            model.head.load_state_dict(self.memory_head)
            model.head.load_state_dict(self.original_head)
            self.memory_temperature = temperature
            self.identity += f'/memory-{digest}/temperature-{temperature}'
        self.compile = lambda request: compile_items(request, compile_request)
        self.prompts = candidate_prompts
        warmup = compile_request("Warmup", {"ready": {"type": "noul", "instructions": "Is the system ready?"}})
        self._score(warmup)

    def prepare(self, request):
        records = self.compile(request)
        sizes = []
        for record in records:
            tokenizer = self.predictor.scorer.model.tokenizer
            lengths = [len(tokenizer.encode(tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))) for p in self.prompts(record)]
            if any(n > 512 for n in lengths):
                raise ValueError("Candidate exceeds the 512-token limit; use a shorter complete source span")
            sizes.extend(lengths)
        return records, sum(sizes), max(sizes)

    def _score(self, records):
        from jev.serving import candidate_batches
        from jev.metrics import softmax
        logits = [[] for _ in records]
        temperatures = [self.predictor.temperature] * len(records)
        if self.memory_head is None:
            groups = [(list(range(len(records))), None, self.predictor.temperature)]
        else:
            groups = [([i for i,r in enumerate(records) if r['kind']=='noul'], self.memory_head, self.memory_temperature),
                      ([i for i,r in enumerate(records) if r['kind']!='noul'], self.original_head, self.predictor.temperature)]
        for indices, head, temperature in groups:
            if not indices:
                continue
            if head is not None:
                self.predictor.scorer.model.head.load_state_dict(head)
            for batch in candidate_batches([records[i] for i in indices], self.batch_size):
                rows, _ = self.predictor.scorer.score([piece for _, piece in batch])
                if len(rows) != len(batch):
                    raise RuntimeError("Unexpected scorer shape")
                for (index, piece), row in zip(batch, rows):
                    expected = 2 if piece["kind"] == "noul" else len(piece["options"])
                    if len(row) != expected or any(not math.isfinite(x) for x in row):
                        raise RuntimeError("Invalid model logits")
                    actual = indices[index]
                    logits[actual].extend(row)
                    temperatures[actual] = temperature
        return [softmax(row, temp) for row,temp in zip(logits,temperatures)]

    def infer(self, groups):
        # Length buckets reduce padding. Software maps rows back to requests/IDs.
        flattened = [(gi, ri, record) for gi, group in enumerate(groups) for ri, record in enumerate(group)]
        flattened.sort(key=lambda row: max(len(p) for p in self.prompts(row[2])))
        scores = self._score([row[2] for row in flattened])
        output = [[None] * len(group) for group in groups]
        for (gi, ri, record), probs in zip(flattened, scores):
            if record["kind"] == "noul":
                decision = {"id": record["id"], "score": probs[1]}
            else:
                index = max(range(len(probs)), key=probs.__getitem__)
                decision = {"id": record["id"], "score": probs[index], "relation": record["answer_keys"][index]}
            output[gi][ri] = decision
        return output
