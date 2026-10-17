"""Shared encoder path (MASTER_PLAN §6): prefix, tokenisation, pooling, normalisation.

Training, evaluation and export all call `embed()`/`encode()` so the contract lives once.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

RECIPES = {
    "e5-base-v2": {
        "model_id": "intfloat/e5-base-v2",
        "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        "prefix": "query: ", "pooling": "mean", "normalize": True, "dim": 768,
    },
    "bge-small-en-v1.5": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "prefix": "", "pooling": "cls", "normalize": True, "dim": 384,
    },
    "e5-small-v2": {
        "model_id": "intfloat/e5-small-v2",
        "revision": "ffb93f3bd4047442299a41ebb6fa998a38507c52",
        "prefix": "query: ", "pooling": "mean", "normalize": True, "dim": 384,
    },
}


def device() -> torch.device:
    return torch.device("xpu" if torch.xpu.is_available() else "cpu")


def add_prefix(texts: list[str], prefix: str) -> list[str]:
    """Apply the prefix exactly once; an input that already carries it is a bug upstream."""
    if prefix and any(t.startswith(prefix) for t in texts):
        raise ValueError(f"input already starts with prefix {prefix!r}")
    return [prefix + t for t in texts]


def load(recipe: dict, path: str | None = None):
    """Tokenizer + model at the pinned revision (or a local fine-tuned directory)."""
    src = path or recipe["model_id"]
    kw = {} if path or not recipe.get("revision") else {"revision": recipe["revision"]}
    tok = AutoTokenizer.from_pretrained(src, **kw)
    model = AutoModel.from_pretrained(src, **kw)
    return tok, model


def pool(last_hidden: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    if how == "mean":
        m = mask.unsqueeze(-1).to(last_hidden.dtype)
        return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
    if how == "cls":
        return last_hidden[:, 0]
    raise ValueError(how)


def tokenize(tok, texts: list[str], recipe: dict, max_length: int) -> dict:
    return tok(add_prefix(texts, recipe["prefix"]), padding=True, truncation=True,
               max_length=max_length, return_tensors="pt")


def embed(model, batch: dict, recipe: dict) -> torch.Tensor:
    """Forward + pooling + optional L2 normalisation. Differentiable (used in training)."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    v = pool(out.last_hidden_state, batch["attention_mask"], recipe["pooling"])
    return F.normalize(v, p=2, dim=-1) if recipe["normalize"] else v


@torch.no_grad()
def encode(model, tok, texts: list[str], recipe: dict, max_length: int, batch_size: int = 64,
           dev: torch.device | None = None) -> np.ndarray:
    """Inference in length-sorted batches; returns float32 rows in input order."""
    dev = dev or next(model.parameters()).device
    model.eval()
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty((len(texts), recipe["dim"]), dtype=np.float32)
    for s in range(0, len(texts), batch_size):
        idx = order[s : s + batch_size]
        b = tokenize(tok, [texts[i] for i in idx], recipe, max_length)
        b = {k: v.to(dev) for k, v in b.items()}
        out[idx] = embed(model, b, recipe).float().cpu().numpy()
    if not np.isfinite(out).all() or (np.linalg.norm(out, axis=1) == 0).any():
        raise ValueError("encoder produced non-finite or zero vectors")
    return out


def token_lengths(tok, texts: list[str], recipe: dict) -> np.ndarray:
    """Untruncated lengths including prefix and special tokens."""
    enc = tok(add_prefix(texts, recipe["prefix"]), truncation=False)
    return np.array([len(x) for x in enc["input_ids"]])


def choose_max_length(lengths: np.ndarray, coverage: float = 0.99, cap: int = 512) -> int:
    """Smallest length L with at least `coverage` of inputs <= L, else the model cap."""
    s = np.sort(np.asarray(lengths))
    need = int(s[int(np.ceil(coverage * len(s))) - 1])
    return min(need, cap)
