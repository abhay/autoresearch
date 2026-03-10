"""
Autoresearch MLX training script. Native Apple Silicon training.
Usage: uv run python train_mlx.py
"""

import os
import gc
import time
import math
from dataclasses import asdict

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from config import (
    GPTConfig, TOTAL_BATCH_SIZE, EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR,
    SCALAR_LR, WEIGHT_DECAY, ADAM_BETAS, DEPTH,
    build_model_config, get_lr_multiplier, get_muon_momentum, get_weight_decay,
)
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, TOKENIZER_DIR, _document_batches

# ---------------------------------------------------------------------------
# Apple Silicon overrides
# ---------------------------------------------------------------------------

DEVICE_BATCH_SIZE = 16
EVAL_TOKENS = 3 * 524288  # smaller eval for faster iteration

# Conservative peak FLOPS for MFU (bf16). Override for your chip.
PEAK_FLOPS = 14.0e12

# ---------------------------------------------------------------------------
# Attention masks
# ---------------------------------------------------------------------------

_mask_cache: dict = {}

def _get_mask(T: int, window_size: int):
    """Return additive attention mask or None for full causal."""
    if window_size <= 0 or window_size >= T:
        return None
    key = (T, window_size)
    if key not in _mask_cache:
        idx = mx.arange(T)
        causal = idx[None, :] > idx[:, None]
        too_far = (idx[:, None] - idx[None, :]) >= window_size
        blocked = causal | too_far
        _mask_cache[key] = mx.where(blocked, mx.array(float("-inf")), mx.array(0.0))
    return _mask_cache[key]

# ---------------------------------------------------------------------------
# Data loading (MLX-native, same best-fit packing as prepare.py)
# ---------------------------------------------------------------------------

def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        doc_buffer.extend(tokenizer.encode(doc_batch, prepend=bos_token))

    while True:
        rows = np.zeros((B, row_capacity), dtype=np.int32)
        for ri in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill()
                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    dl = len(doc)
                    if dl <= remaining and dl > best_len:
                        best_idx, best_len = i, dl
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    rows[ri, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    si = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(si)
                    rows[ri, pos:pos + remaining] = doc[:remaining]
                    pos += remaining
        yield mx.array(rows[:, :-1]), mx.array(rows[:, 1:]), epoch

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_token_bytes = None

def evaluate_bpb(model, tokenizer, batch_size):
    global _token_bytes
    if _token_bytes is None:
        import torch
        tb = torch.load(os.path.join(TOKENIZER_DIR, "token_bytes.pt"), map_location="cpu")
        _token_bytes = mx.array(tb.numpy())

    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="none").reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = _token_bytes[y_flat]
        mask = (nbytes > 0).astype(mx.float32)
        sn = (loss * mask).sum()
        sb = nbytes.sum()
        mx.eval(sn, sb)
        total_nats += sn.item()
        total_bytes += sb.item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer) else None
        )
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=10000)
        self._scale = self.head_dim ** -0.5

    def __call__(self, x, ve, mask):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer)
        if ve is not None and self.ve_gate is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        # Transpose to (B, H, T, D) for RoPE and SDPA
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = norm(self.rope(q))
        k = norm(self.rope(k))

        if mask is None:
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self._scale, mask="causal")
        else:
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self._scale, mask=mask)

        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2  # squared ReLU
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, mask):
        x = x + self.attn(norm(x), ve, mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas = mx.zeros((config.n_layer,))

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        }

        # Precompute masks per layer
        self._masks = []
        for ws in self._compute_window_sizes(config):
            w = ws[0]
            self._masks.append(_get_mask(config.sequence_len, w))

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w = config.sequence_len
        short_w = long_w // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def init_weights(self):
        s = 3**0.5 * self.config.n_embd**-0.5
        self.wte.weight = mx.random.normal(self.wte.weight.shape).astype(mx.bfloat16)
        self.lm_head.weight = (mx.random.normal(self.lm_head.weight.shape) * 0.001).astype(mx.bfloat16)

        for block in self.blocks:
            for layer in [block.attn.c_q, block.attn.c_k, block.attn.c_v, block.mlp.c_fc]:
                layer.weight = mx.random.uniform(-s, s, layer.weight.shape).astype(mx.bfloat16)
            block.attn.c_proj.weight = mx.zeros(block.attn.c_proj.weight.shape, dtype=mx.bfloat16)
            block.mlp.c_proj.weight = mx.zeros(block.mlp.c_proj.weight.shape, dtype=mx.bfloat16)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight = mx.zeros(block.attn.ve_gate.weight.shape, dtype=mx.bfloat16)

        self.resid_lambdas = mx.ones((self.config.n_layer,))
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1)

        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-s, s, ve.weight.shape).astype(mx.bfloat16)

    def estimate_flops(self):
        nparams = sum(p.size for _, p in tree_flatten(self.parameters()))
        ve_numel = sum(ve.weight.size for ve in self.value_embeds.values())
        exclude = self.wte.weight.size + ve_numel + self.resid_lambdas.size + self.x0_lambdas.size
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for ws in self._compute_window_sizes(self.config):
            w = ws[0]
            eff = t if w < 0 else min(w, t)
            attn_flops += 12 * h * q * eff
        return 6 * (nparams - exclude) + attn_flops

    def __call__(self, idx):
        B, T = idx.shape
        x = norm(self.wte(idx))
        x0 = x
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, self._masks[i])
        x = norm(x)
        logits = self.lm_head(x).astype(mx.float32)
        return 15.0 * mx.tanh(logits / 15.0)

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW for MLX)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def _apply_updates(model, updates):
    """Apply flat {dotted.path: value} updates, respecting model's list/dict/module structure."""
    for path, val in updates.items():
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        key = parts[-1]
        if isinstance(obj, dict):
            obj[key] = val
        elif isinstance(obj, list):
            obj[int(key)] = val
        else:
            setattr(obj, key, val)


class MuonAdamW:
    """Multi-group optimizer: AdamW for embeddings/scalars, Muon for 2D matrices."""

    def __init__(self, groups):
        """groups: list of dicts with 'kind', 'filter', 'lr', etc."""
        self.groups = groups
        for g in self.groups:
            g["initial_lr"] = g["lr"]
        self.state = {}
        self._assignments = None  # lazily built on first step

    def _assign(self, flat_params):
        self._assignments = {}
        for path, param in flat_params:
            for i, g in enumerate(self.groups):
                if g["filter"](path, param):
                    self._assignments[path] = i
                    break

    @property
    def state_arrays(self):
        out = []
        for v in self.state.values():
            if isinstance(v, dict):
                out.extend(a for a in v.values() if isinstance(a, mx.array))
            elif isinstance(v, mx.array):
                out.append(v)
        return out

    def step(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        flat_params_list = tree_flatten(model.trainable_parameters())
        flat_params = dict(flat_params_list)

        if self._assignments is None:
            self._assign(flat_params_list)

        # Bucket by group
        buckets = {}
        for path, grad in flat_grads.items():
            idx = self._assignments.get(path)
            if idx is not None:
                buckets.setdefault(idx, []).append((path, flat_params[path], grad))

        updates = {}
        for idx, items in buckets.items():
            g = self.groups[idx]
            if g["kind"] == "adamw":
                for path, param, grad in items:
                    updates[path] = self._adamw(path, param, grad, g)
            elif g["kind"] == "muon":
                self._muon(items, g, updates)

        _apply_updates(model, updates)

    def _adamw(self, path, param, grad, g):
        s = self.state.setdefault(path, {"step": 0, "m": mx.zeros_like(param), "v": mx.zeros_like(param)})
        s["step"] += 1
        lr, (b1, b2), eps, wd = g["lr"], g["betas"], g["eps"], g["weight_decay"]
        param = param * (1 - lr * wd)
        s["m"] = b1 * s["m"] + (1 - b1) * grad
        s["v"] = b2 * s["v"] + (1 - b2) * (grad * grad)
        bc1 = 1 - b1 ** s["step"]
        bc2 = 1 - b2 ** s["step"]
        param = param - (lr / bc1) * s["m"] / (mx.sqrt(s["v"] / bc2) + eps)
        return param

    def _muon(self, items, g, updates):
        by_shape = {}
        for path, param, grad in items:
            by_shape.setdefault(param.shape, []).append((path, param, grad))

        for shape, shape_items in by_shape.items():
            paths = [p for p, _, _ in shape_items]
            stacked_g = mx.stack([gr for _, _, gr in shape_items])
            stacked_p = mx.stack([p for _, p, _ in shape_items])

            sk = tuple(sorted(paths))
            s = self.state.setdefault(sk, {})
            num = len(shape_items)
            if "mom" not in s:
                s["mom"] = mx.zeros_like(stacked_g)
            if "smom" not in s:
                sm_shape = (num, shape[-2], 1) if shape[-2] >= shape[-1] else (num, 1, shape[-1])
                s["smom"] = mx.zeros(sm_shape, dtype=stacked_p.dtype)

            momentum = g["momentum"]
            lr = g["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
            wd = g["weight_decay"]
            beta2 = g.get("beta2", 0.95)
            ns_steps = g.get("ns_steps", 5)
            red_dim = -1 if shape[-2] >= shape[-1] else -2

            # Nesterov momentum
            s["mom"] = momentum * s["mom"] + (1 - momentum) * stacked_g
            ng = (1 - momentum) * stacked_g + momentum * s["mom"]

            # Polar Express orthogonalization (float32 — Metal bfloat16 matmul lacks float32 accumulation)
            X = ng.astype(mx.float32)
            X = X / (mx.sqrt(mx.sum(X * X, axis=(-2, -1), keepdims=True)) * 1.02 + 1e-6)
            ndim = X.ndim
            swap = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
            if shape[-2] > shape[-1]:
                for a, b, c in polar_express_coeffs[:ns_steps]:
                    A = mx.transpose(X, swap) @ X
                    B_mat = b * A + c * (A @ A)
                    X = a * X + X @ B_mat
            else:
                for a, b, c in polar_express_coeffs[:ns_steps]:
                    A = X @ mx.transpose(X, swap)
                    B_mat = b * A + c * (A @ A)
                    X = a * X + B_mat @ X
            ng = X

            # NorMuon variance reduction
            v_mean = mx.mean(ng.astype(mx.float32) ** 2, axis=red_dim, keepdims=True)
            rds = ng.shape[red_dim]
            v_norm = mx.sqrt(mx.sum(v_mean, axis=(-2, -1), keepdims=True) * rds)
            s["smom"] = beta2 * s["smom"] + (1 - beta2) * v_mean.astype(s["smom"].dtype)
            step_size = mx.rsqrt(mx.maximum(s["smom"], 1e-10))
            scaled = (v_mean * rds) * (step_size.astype(mx.float32) ** 2)
            v_norm_new = mx.sqrt(mx.sum(scaled, axis=(-2, -1), keepdims=True))
            final_scale = step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))
            ng = ng * final_scale.astype(ng.dtype)

            # Cautious weight decay + update
            cmask = ((ng * stacked_p) >= 0).astype(ng.dtype)
            stacked_p = stacked_p - lr * ng - lr * wd * stacked_p * cmask

            for i, (path, _, _) in enumerate(shape_items):
                updates[path] = stacked_p[i]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

config = build_model_config(DEPTH, vocab_size, MAX_SEQ_LEN)
print(f"Model config: {asdict(config)}")

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())

num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
num_flops_per_token = model.estimate_flops()
print(f"Parameters: {num_params / 1e6:.1f}M")
print(f"FLOPs/token: {num_flops_per_token:.2e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

model_dim = config.n_embd
dmodel_lr_scale = (model_dim / 768) ** -0.5
print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

optimizer = MuonAdamW([
    dict(kind="adamw", filter=lambda p, v: "lm_head" in p,
         lr=UNEMBEDDING_LR * dmodel_lr_scale, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
    dict(kind="adamw", filter=lambda p, v: "wte" in p,
         lr=EMBEDDING_LR * dmodel_lr_scale, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
    dict(kind="adamw", filter=lambda p, v: "value_embeds" in p,
         lr=EMBEDDING_LR * dmodel_lr_scale, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
    dict(kind="adamw", filter=lambda p, v: "resid_lambdas" in p,
         lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
    dict(kind="adamw", filter=lambda p, v: "x0_lambdas" in p,
         lr=SCALAR_LR, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    dict(kind="muon", filter=lambda p, v: v.ndim >= 2,
         lr=MATRIX_LR, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY),
])

def loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    t0 = time.time()

    # Gradient accumulation
    accum_grads = None
    accum_loss = mx.array(0.0)
    for micro_step in range(grad_accum_steps):
        loss, grads = loss_and_grad_fn(model, x, y)
        mx.eval(loss, grads)
        accum_loss = accum_loss + loss
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
        x, y, epoch = next(train_loader)

    if grad_accum_steps > 1:
        accum_grads = tree_map(lambda g: g * (1.0 / grad_accum_steps), accum_grads)
    train_loss_f = (accum_loss / grad_accum_steps).item()

    # Fast fail
    if train_loss_f > 100:
        print("FAIL")
        exit(1)

    # Schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_wd = get_weight_decay(progress)
    for g in optimizer.groups:
        g["lr"] = g["initial_lr"] * lrm
        if g["kind"] == "muon":
            g["momentum"] = muon_momentum
            g["weight_decay"] = muon_wd

    optimizer.step(model, accum_grads)
    mx.eval(model.parameters(), *optimizer.state_arrays)

    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct = 100 * progress
    tok_s = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct:.1f}%) | loss: {debiased:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_s:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Final eval
# ---------------------------------------------------------------------------

val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()
startup_time = t_start_training - t_start
steady_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / PEAK_FLOPS if total_training_time > 0 else 0
peak_mem = mx.get_peak_memory() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_mem:.1f}")
print(f"mfu_percent:      {steady_mfu:.2f}")
print(f"total_tokens_M:   {step * TOTAL_BATCH_SIZE / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
