#!/usr/bin/env python3
"""
ZRO Intelligence — Compiler v0.2 (Auto-save + MMLU ready)
"""

import argparse
import gc
import math
import os
import sys
import time
import json
from datetime import datetime
from typing import Optional, Tuple

def check_dependencies():
    missing = []
    for pkg, import_name in [("torch", "torch"), ("transformers", "transformers"), ("datasets", "datasets")]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print(f"Run: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ZRO ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

class ZROAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        chunk_size: int = 256,
        eps: float = 1e-6,
        gate_bias: float = 2.0,
        use_hedgehog: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
        self.eps = eps
        self.use_hedgehog = use_hedgehog
        self.feature_dim = 2 * self.head_dim if use_hedgehog else self.head_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        if use_hedgehog:
            self.W1 = nn.Parameter(
                torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02
            )

        self.g_proj = nn.Linear(d_model, d_model, bias=True)
        self._init_weights(gate_bias)

    def _init_weights(self, gate_bias: float):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)

        if self.use_hedgehog:
            for h in range(self.num_heads):
                eye = torch.eye(self.head_dim) * 0.1
                self.W1.data[h] = eye + torch.randn_like(eye) * 0.01

        nn.init.normal_(self.g_proj.weight, std=0.01)
        nn.init.constant_(self.g_proj.bias, gate_bias)

    def init_from_pretrained_gpt2(self, gpt2_attn):
        with torch.no_grad():
            w = gpt2_attn.c_attn.weight
            b = gpt2_attn.c_attn.bias
            D = self.d_model

            self.q_proj.weight.copy_(w[:, :D].T)
            self.k_proj.weight.copy_(w[:, D : 2 * D].T)
            self.v_proj.weight.copy_(w[:, 2 * D :].T)
            if b is not None:
                self.q_proj.bias.copy_(b[:D])
                self.k_proj.bias.copy_(b[D : 2 * D])
                self.v_proj.bias.copy_(b[2 * D :])

            self.out_proj.weight.copy_(gpt2_attn.c_proj.weight.T)
            if gpt2_attn.c_proj.bias is not None:
                self.out_proj.bias.copy_(gpt2_attn.c_proj.bias)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_hedgehog:
            wx = torch.einsum("hij,bhnj->bhni", self.W1, x)
            wx = wx.clamp(min=-20, max=20)
            pos = torch.exp(wx)
            neg = torch.exp(-wx)
            return torch.cat([pos, neg], dim=-1)
        else:
            return F.elu(x) + 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, N, D = hidden_states.shape
        H, Hd = self.num_heads, self.head_dim
        fd = self.feature_dim
        C = self.chunk_size
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).reshape(B, N, H, Hd).transpose(1, 2)
        K = self.k_proj(hidden_states).reshape(B, N, H, Hd).transpose(1, 2)
        V = self.v_proj(hidden_states).reshape(B, N, H, Hd).transpose(1, 2)
        G = torch.sigmoid(self.g_proj(hidden_states)).reshape(B, N, H, Hd).transpose(1, 2)

        Q = self.phi(Q)
        K = self.phi(K)

        if self.use_hedgehog:
            G = torch.cat([G, G], dim=-1)

        if attention_mask is not None:
            mask = (attention_mask > -1000).float()
            mask_len = mask.shape[-1]
            if mask_len > N:
                mask = mask[..., -N:]
            mask = mask.squeeze(1).squeeze(1).unsqueeze(1).unsqueeze(-1)
            K = K * mask
            V = V * mask

        if layer_past is not None:
            S_prev, z_prev = layer_past
        else:
            S_prev = torch.zeros(B, H, fd, Hd, device=device, dtype=dtype)
            z_prev = torch.zeros(B, H, fd, device=device, dtype=dtype)

        outputs = []
        for start in range(0, N, C):
            end = min(start + C, N)
            Ci = end - start

            Qc = Q[:, :, start:end, :]
            Kc = K[:, :, start:end, :]
            Vc = V[:, :, start:end, :]
            Gc = G[:, :, start:end, :]

            log_Gc = torch.cumsum(torch.log(Gc.clamp(min=1e-6)), dim=2)
            G_cum = torch.exp(log_Gc)

            kv = torch.einsum("bhci,bhcj->bhcij", Kc, Vc)
            kv_weighted = kv / (G_cum.unsqueeze(-1) + 1e-6)
            kv_cumsum = torch.cumsum(kv_weighted, dim=2)

            S_total = S_prev.unsqueeze(2) + kv_cumsum
            Q_weighted = Qc * G_cum
            num = torch.einsum("bhci,bhcij->bhcj", Q_weighted, S_total)

            k_weighted = Kc / (G_cum + 1e-6)
            z_cumsum = torch.cumsum(k_weighted, dim=2)
            z_total = z_prev.unsqueeze(2) + z_cumsum
            den = (
                torch.einsum("bhci,bhci->bhc", Q_weighted, z_total)
                .unsqueeze(-1)
                .clamp(min=self.eps)
            )

            out_chunk = num / den
            outputs.append(out_chunk)

            if end < N or use_cache:
                S_prev = G_cum[:, :, -1, :].unsqueeze(-1) * S_total[:, :, -1, :, :]
                z_prev = G_cum[:, :, -1, :] * z_total[:, :, -1, :]

        out = torch.cat(outputs, dim=2)

        if head_mask is not None:
            out = out * head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)

        present = None
        if use_cache:
            present = (S_prev, z_prev)

        if output_attentions:
            return out, present, None
        return out, present


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ZRO COMPILER v0.2
# ══════════════════════════════════════════════════════════════════════════════

class ZROCompiler:
    SUPPORTED_MODELS = {
        "gpt2": {"arch": "gpt2", "d_model": 768, "heads": 12},
        "gpt2-medium": {"arch": "gpt2", "d_model": 1024, "heads": 16},
    }

    def __init__(
        self,
        model_name: str,
        chunk_size: int = 256,
        device: Optional[str] = None,
        verbose: bool = True,
        use_hedgehog: bool = False,
    ):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.config = self.SUPPORTED_MODELS[model_name]
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_hedgehog = use_hedgehog
        self.report_lines = []
        self.results = {}
        self.zro_layers = None
        self.original_ppl = None
        self.final_ppl = None
        self.student_model = None

        self._log(f"ZRO Compiler v0.2")
        self._log(f"Model:     {model_name}")
        self._log(f"Hedgehog:  {use_hedgehog}")
        self._log(f"Device:    {self.device}")
        self._log(f"Chunk:     {chunk_size} tokens")
        self._log("─" * 50)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
        self.report_lines.append(msg)

    def load(self):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self._log("\n[1/5] Loading pretrained model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.original_model = (
            GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device).eval()
        )
        for p in self.original_model.parameters():
            p.requires_grad = False

        self.num_layers = len(self.original_model.transformer.h)
        n_params = sum(p.numel() for p in self.original_model.parameters())
        self._log(
            f"  Loaded: {self.model_name} | {n_params:,} params | {self.num_layers} layers"
        )
        return self

    def compile(self, pretrained_init: bool = True):
        self._log("\n[2/5] Compiling: replacing attention with ZRO...")

        d_model = self.config["d_model"]
        heads = self.config["heads"]

        self.zro_layers = nn.ModuleList(
            [
                ZROAttention(
                    d_model=d_model,
                    num_heads=heads,
                    chunk_size=self.chunk_size,
                    use_hedgehog=self.use_hedgehog,
                )
                for _ in range(self.num_layers)
            ]
        ).to(self.device)

        if pretrained_init:
            self._log("  Initializing from pretrained Q/K/V weights...")
            for i in range(self.num_layers):
                src = self.original_model.transformer.h[i].attn
                self.zro_layers[i].init_from_pretrained_gpt2(src)

        trainable = sum(p.numel() for p in self.zro_layers.parameters())
        self._log(f"  ZRO layers: {self.num_layers}")
        self._log(f"  Trainable:  {trainable:,}")
        self._log(f"  Pretrained: {pretrained_init}")

        self.results["trainable_params"] = trainable
        self.results["pretrained_init"] = pretrained_init
        return self

    def benchmark_layer(self, label: str = ""):
        self._log(
            f"\n[3/5] Layer-level benchmark {f'({label})' if label else ''}..."
        )

        if not torch.cuda.is_available():
            self._log("  CUDA not available — skipping")
            return self

        d_model = self.config["d_model"]
        heads = self.config["heads"]

        class CausalAttn(nn.Module):
            def __init__(self, dm, nh):
                super().__init__()
                self.nh, self.hd = nh, dm // nh
                self.sc = (dm // nh) ** -0.5
                self.c_attn = nn.Linear(dm, 3 * dm, bias=True)
                self.c_proj = nn.Linear(dm, dm, bias=True)

            def forward(self, x):
                B, N, D = x.shape
                qkv = self.c_attn(x)
                Q, K, V = qkv.split(D, -1)
                Q = Q.reshape(B, N, self.nh, self.hd).transpose(1, 2)
                K = K.reshape(B, N, self.nh, self.hd).transpose(1, 2)
                V = V.reshape(B, N, self.nh, self.hd).transpose(1, 2)
                sc = (Q @ K.transpose(-2, -1)) * self.sc
                mask = torch.triu(torch.ones(N, N, device=x.device), 1).bool()
                sc = sc.masked_fill(mask[None, None], float("-inf"))
                out = torch.softmax(sc, -1) @ V
                return self.c_proj(out.transpose(1, 2).reshape(B, N, D))

        attn_ref = CausalAttn(d_model, heads).to(self.device).eval()
        zro_ref = ZROAttention(
            d_model, heads, self.chunk_size, use_hedgehog=self.use_hedgehog
        ).to(self.device).eval()

        def measure_layer(model, seq_len, runs=10, warmup=5):
            x = torch.randn(1, seq_len, d_model, device=self.device)
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(x)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.no_grad():
                for _ in range(runs):
                    out = model(x)
            torch.cuda.synchronize()
            elapsed = (time.time() - t0) / runs
            mem = torch.cuda.max_memory_allocated() / 1e6
            del x
            torch.cuda.empty_cache()
            gc.collect()
            return mem, seq_len / elapsed

        lengths = [512, 1024, 2048, 4096, 8192]
        mem_key = f"layer_memory_{label}" if label else "layer_memory"

        self._log(
            f"\n  {'Tokens':>8} | {'Attn MB':>11} {'ZRO MB':>10} {'Saved':>7} | "
            f"{'Attn tok/s':>12} {'ZRO tok/s':>11} {'Ratio':>7}"
        )
        self._log("  " + "─" * 78)

        mem_results = []
        for N in lengths:
            try:
                a_mem, a_spd = measure_layer(attn_ref, N)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                a_mem, a_spd = float("inf"), 0

            z_mem, z_spd = measure_layer(zro_ref, N)

            if a_mem != float("inf"):
                saved = (a_mem - z_mem) / a_mem * 100
                ratio = z_spd / a_spd if a_spd > 0 else 0
                line = (
                    f"  {N:>8} | {a_mem:>10.1f}  {z_mem:>9.1f}  {saved:>6.1f}% | "
                    f"{a_spd:>12,.0f} {z_spd:>11,.0f} {ratio:>6.2f}x"
                )
            else:
                line = (
                    f"  {N:>8} | {'OOM':>10}  {z_mem:>9.1f}  {'N/A':>6} | "
                    f"{'OOM':>12} {z_spd:>11,.0f} {'N/A':>6}"
                )

            self._log(line)
            mem_results.append(
                {
                    "tokens": N,
                    "attn_mb": a_mem,
                    "zro_mb": z_mem,
                    "saved_pct": saved if a_mem != float("inf") else None,
                    "attn_toks": a_spd,
                    "zro_toks": z_spd,
                }
            )

        self.results[mem_key] = mem_results
        return self

    def benchmark_full_model(self, label: str = ""):
        self._log(
            f"\n[4/5] Full-model benchmark {f'({label})' if label else ''}..."
        )

        if not torch.cuda.is_available():
            self._log("  CUDA not available — skipping")
            return self

        try:
            from transformers import GPT2LMHeadModel
        except ImportError:
            self._log("  transformers not available — skipping")
            return self

        orig_model = GPT2LMHeadModel.from_pretrained(self.model_name).to(
            self.device
        ).eval()
        for p in orig_model.parameters():
            p.requires_grad = False

        student = self._build_student_model()

        def measure_model(model, seq_len, runs=5, warmup=3):
            x = torch.randint(0, 50257, (1, seq_len), device=self.device)
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(x).logits
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.no_grad():
                for _ in range(runs):
                    _ = model(x).logits
            torch.cuda.synchronize()
            elapsed = (time.time() - t0) / runs
            mem = torch.cuda.max_memory_allocated() / 1e6
            del x
            torch.cuda.empty_cache()
            gc.collect()
            return mem, seq_len / elapsed

        lengths = [128, 256, 512, 1024]
        mem_key = f"full_memory_{label}" if label else "full_memory"

        self._log(
            f"\n  {'Tokens':>8} | {'Orig MB':>11} {'ZRO MB':>10} {'Saved':>7} | "
            f"{'Orig tok/s':>12} {'ZRO tok/s':>11}"
        )
        self._log("  " + "─" * 72)

        mem_results = []
        for N in lengths:
            o_mem, o_spd = measure_model(orig_model, N)
            z_mem, z_spd = measure_model(student, N)
            saved = (o_mem - z_mem) / o_mem * 100 if o_mem > 0 else 0
            self._log(
                f"  {N:>8} | {o_mem:>10.1f}  {z_mem:>9.1f}  {saved:>6.1f}% | "
                f"{o_spd:>12,.0f} {z_spd:>11,.0f}"
            )
            mem_results.append(
                {
                    "tokens": N,
                    "orig_mb": o_mem,
                    "zro_mb": z_mem,
                    "saved_pct": saved,
                    "orig_toks": o_spd,
                    "zro_toks": z_spd,
                }
            )

        self.results[mem_key] = mem_results
        del orig_model, student
        torch.cuda.empty_cache()
        gc.collect()
        return self

    def _evaluate_ppl(self, model, chunks, batch_size=4):
        model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(chunks) - batch_size, batch_size):
                ids = torch.tensor(chunks[i : i + batch_size], device=self.device)
                out = model(ids, labels=ids)
                total_loss += out.loss.item()
                n += 1
        avg = total_loss / max(n, 1)
        return avg, math.exp(avg)

    def _tokenize(self, texts, block_size=256):
        text = "\n\n".join(t for t in texts if t.strip())
        toks = self.tokenizer.encode(text)
        return [
            toks[i : i + block_size]
            for i in range(0, len(toks) - block_size, block_size)
        ]

    def _build_student_model(self):
        from transformers import GPT2LMHeadModel

        student = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.zro_layers.eval()
        for i in range(self.num_layers):
            layer = self.zro_layers[i]

            def make_fwd(l):
                def fwd(
                    hidden_states,
                    layer_past=None,
                    attention_mask=None,
                    head_mask=None,
                    use_cache=False,
                    output_attentions=False,
                    **kwargs,
                ):
                    out, present = l(
                        hidden_states,
                        layer_past=layer_past,
                        attention_mask=attention_mask,
                        head_mask=head_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )
                    return out, present

                return fwd

            student.transformer.h[i].attn.forward = make_fwd(layer)

        return student.to(self.device)

    def train(
        self,
        distill_epochs: int = 3,
        finetune_epochs: int = 2,
        distill_lr: float = 3e-5,
        finetune_lr: float = 5e-6,
        batch_size: int = 4,
        block_size: int = 256,
        grad_clip: float = 1.0,
        log_every: int = 100,
        freeze_non_hedgehog: bool = False,
        dataset_name: str = "wikitext",
    ):
        self._log("\n[5/5] Training...")

        if self.zro_layers is None:
            raise RuntimeError("Call compile() before train()")

        self._log(f"  Loading {dataset_name} dataset...")
        try:
            from datasets import load_dataset

            if dataset_name == "wikitext":
                ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
                train_chunks = self._tokenize(ds["train"]["text"], block_size)
                eval_chunks = self._tokenize(ds["validation"]["text"], block_size)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        except Exception as e:
            self._log(f"  ERROR loading dataset: {e}")
            return self

        self._log(
            f"  Train chunks: {len(train_chunks):,} | Eval: {len(eval_chunks):,}"
        )

        self._log("  Evaluating original model...")
        orig_loss, orig_ppl = self._evaluate_ppl(
            self.original_model, eval_chunks, batch_size
        )
        self.original_ppl = orig_ppl
        self.results["original_ppl"] = orig_ppl
        self.results["original_loss"] = orig_loss
        self._log(f"  Original {self.model_name} PPL: {orig_ppl:.2f}")

        if distill_epochs > 0:
            self._log(f"\n  STAGE 1: Distillation ({distill_epochs} epochs)...")

            if freeze_non_hedgehog and self.use_hedgehog:
                self._log("  Freezing non-Hedgehog parameters...")
                for p in self.zro_layers.parameters():
                    p.requires_grad = False
                for layer in self.zro_layers:
                    layer.W1.requires_grad = True

            cap_in, cap_out = {}, {}
            hooks = []
            for i in range(self.num_layers):

                def pre(mod, args, idx=i):
                    cap_in[idx] = args[0].detach()

                def post(mod, args, out, idx=i):
                    cap_out[idx] = out[0].detach()

                hooks.append(
                    self.original_model.transformer.h[i].attn.register_forward_pre_hook(
                        pre
                    )
                )
                hooks.append(
                    self.original_model.transformer.h[i].attn.register_forward_hook(
                        post
                    )
                )

            opt = torch.optim.AdamW(
                self.zro_layers.parameters(), lr=distill_lr, weight_decay=0.01
            )
            self.zro_layers.train()
            step = skipped = 0
            t0 = time.time()

            for epoch in range(distill_epochs):
                ep_loss, ep_n = 0.0, 0
                for i in range(0, len(train_chunks) - batch_size, batch_size):
                    ids = torch.tensor(
                        train_chunks[i : i + batch_size], device=self.device
                    )
                    cap_in.clear()
                    cap_out.clear()
                    with torch.no_grad():
                        self.original_model(ids)

                    loss = torch.tensor(0.0, device=self.device)
                    for li in range(self.num_layers):
                        pred, _ = self.zro_layers[li](cap_in[li])
                        tgt = cap_out[li]
                        var = tgt.var(dim=[0, 1], keepdim=True).clamp(min=1e-8)
                        loss = loss + ((pred - tgt) ** 2 / var).mean()
                    loss = loss / self.num_layers

                    if not torch.isfinite(loss):
                        skipped += 1
                        opt.zero_grad()
                        continue

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.zro_layers.parameters(), grad_clip
                    )
                    opt.step()

                    ep_loss += loss.item()
                    ep_n += 1
                    step += 1
                    if step % log_every == 0:
                        gate_val = (
                            torch.sigmoid(self.zro_layers[0].g_proj.bias)
                            .mean()
                            .item()
                        )
                        self._log(
                            f"    Ep{epoch+1} Step{step:>5} | "
                            f"Loss {loss.item():.5f} | Gate {gate_val:.3f} | "
                            f"{(time.time()-t0)/60:.1f}min"
                        )

                avg = ep_loss / max(ep_n, 1)
                self._log(f"  Epoch {epoch+1} | Avg distill loss: {avg:.5f}")

            for h in hooks:
                h.remove()
            self.results["distill_time_min"] = (time.time() - t0) / 60
            self.results["distill_epochs"] = distill_epochs

            self._log("  Evaluating post-distill...")
            student = self._build_student_model()
            _, ppl_distill = self._evaluate_ppl(student, eval_chunks, batch_size)
            self.results["ppl_post_distill"] = ppl_distill
            self._log(
                f"  Post-distill PPL: {ppl_distill:.2f}  "
                f"(gap {(ppl_distill-orig_ppl)/orig_ppl*100:+.1f}%)"
            )
            del student
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if finetune_epochs > 0:
            self._log(f"\n  STAGE 2: LM Fine-tuning ({finetune_epochs} epochs)...")

            student = self._build_student_model()
            for p in student.parameters():
                p.requires_grad = False
            self.zro_layers.train()
            for p in self.zro_layers.parameters():
                p.requires_grad = True

            opt2 = torch.optim.AdamW(
                self.zro_layers.parameters(), lr=finetune_lr, weight_decay=0.01
            )
            student.train()
            step2 = 0
            t2 = time.time()

            for epoch in range(finetune_epochs):
                ep_loss, ep_n = 0.0, 0
                for i in range(0, len(train_chunks) - batch_size, batch_size):
                    ids = torch.tensor(
                        train_chunks[i : i + batch_size], device=self.device
                    )
                    loss = student(ids, labels=ids).loss
                    if not torch.isfinite(loss):
                        opt2.zero_grad()
                        continue

                    opt2.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.zro_layers.parameters(), grad_clip * 0.5
                    )
                    opt2.step()
                    ep_loss += loss.item()
                    ep_n += 1
                    step2 += 1

                    if step2 % log_every == 0:
                        run_ppl = math.exp(ep_loss / max(ep_n, 1))
                        self._log(
                            f"    Ep{epoch+1} Step{step2:>5} | "
                            f"Loss {loss.item():.4f} | RunPPL {run_ppl:.1f} | "
                            f"{(time.time()-t2)/60:.1f}min"
                        )

                avg = ep_loss / max(ep_n, 1)
                self._log(
                    f"  Epoch {epoch+1} | Avg LM loss: {avg:.4f} | "
                    f"PPL: {math.exp(avg):.2f}"
                )

            self._log("  Final perplexity evaluation...")
            student.eval()
            final_loss, final_ppl = self._evaluate_ppl(
                student, eval_chunks, batch_size
            )
            self.final_ppl = final_ppl
            self.results["final_ppl"] = final_ppl
            self.results["final_loss"] = final_loss
            self.results["finetune_time_min"] = (time.time() - t2) / 60
            self.results["finetune_epochs"] = finetune_epochs
            self._log(
                f"  Final PPL: {final_ppl:.2f}  "
                f"(gap {(final_ppl-orig_ppl)/orig_ppl*100:+.1f}%)"
            )

            # AUTO-SAVE: Store student model for MMLU and future use
            self._log("\n  Auto-saving compiled model...")
            self.student_model = student
            save_path = "./zro-gpt2-compiled"
            student.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self._log(f"  Saved to {save_path}")

    def save_report(self, path: str):
        report = self._build_report()
        with open(path, "w") as f:
            f.write(report)
        print(f"\nReport saved: {path}")
        return path

    def _build_report(self) -> str:
        lines = []
        W = 70
        lines.append("=" * W)
        lines.append("ZRO INTELLIGENCE — COMPILATION REPORT v0.2")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * W)

        lines.append(f"\nMODEL:   {self.model_name}")
        lines.append(f"HEDGEHOG: {self.use_hedgehog}")
        lines.append(f"DEVICE:  {self.device}")
        lines.append(f"CHUNK:   {self.chunk_size} tokens")

        if self.results.get("trainable_params"):
            lines.append(f"ZRO PARAMS: {self.results['trainable_params']:,}")

        if self.results.get("original_ppl"):
            lines.append(f"\n{'─'*W}")
            lines.append("LANGUAGE MODEL QUALITY (WikiText-2 Validation PPL)")
            lines.append(f"{'─'*W}")
            orig = self.results["original_ppl"]
            lines.append(f"  Original {self.model_name}: {orig:.2f}")
            if self.results.get("ppl_post_distill"):
                pd = self.results["ppl_post_distill"]
                lines.append(
                    f"  After Stage 1 distillation: {pd:.2f}  "
                    f"(gap {(pd-orig)/orig*100:+.1f}%)"
                )
            if self.results.get("final_ppl"):
                fp = self.results["final_ppl"]
                lines.append(
                    f"  After Stage 2 fine-tuning:  {fp:.2f}  "
                    f"(gap {(fp-orig)/orig*100:+.1f}%)"
                )

        for key in ["layer_memory", "layer_memory_before", "layer_memory_after"]:
            if key in self.results:
                label = key.replace("layer_memory_", "").upper() if "_" in key else ""
                lines.append(f"\n{'─'*W}")
                lines.append(f"LAYER-LEVEL MEMORY BENCHMARK {f'({label})' if label else ''}")
                lines.append(f"{'─'*W}")
                lines.append(
                    f"  {'Tokens':>8} | {'Attn MB':>11} {'ZRO MB':>10} {'Saved':>7} | "
                    f"{'Attn tok/s':>12} {'ZRO tok/s':>11}"
                )
                lines.append("  " + "─" * 66)
                for r in self.results[key]:
                    if r["saved_pct"] is not None:
                        line = (
                            f"  {r['tokens']:>8} | {r['attn_mb']:>10.1f}  "
                            f"{r['zro_mb']:>9.1f}  {r['saved_pct']:>6.1f}% | "
                            f"{r['attn_toks']:>12,.0f} {r['zro_toks']:>11,.0f}"
                        )
                    else:
                        line = (
                            f"  {r['tokens']:>8} | {'OOM':>11}  "
                            f"{r['zro_mb']:>9.1f}  {'N/A':>6} | "
                            f"{'OOM':>12} {r['zro_toks']:>11,.0f}"
                        )
                    lines.append(line)

        for key in ["full_memory", "full_memory_before", "full_memory_after"]:
            if key in self.results:
                label = key.replace("full_memory_", "").upper() if "_" in key else ""
                lines.append(f"\n{'─'*W}")
                lines.append(f"FULL MODEL MEMORY BENCHMARK {f'({label})' if label else ''}")
                lines.append(f"{'─'*W}")
                lines.append(
                    f"  {'Tokens':>8} | {'Orig MB':>11} {'ZRO MB':>10} {'Saved':>7} | "
                    f"{'Orig tok/s':>12} {'ZRO tok/s':>11}"
                )
                lines.append("  " + "─" * 66)
                for r in self.results[key]:
                    lines.append(
                        f"  {r['tokens']:>8} | {r['orig_mb']:>10.1f}  "
                        f"{r['zro_mb']:>9.1f}  {r['saved_pct']:>6.1f}% | "
                        f"{r['orig_toks']:>12,.0f} {r['zro_toks']:>11,.0f}"
                    )

        lines.append(f"\n{'─'*W}")
        lines.append("TRAINING SUMMARY")
        lines.append(f"{'─'*W}")
        if self.results.get("distill_epochs"):
            t = self.results.get("distill_time_min", 0)
            lines.append(
                f"  Stage 1 distillation: {self.results['distill_epochs']} epochs, {t:.1f} min"
            )
        if self.results.get("finetune_epochs"):
            t = self.results.get("finetune_time_min", 0)
            lines.append(
                f"  Stage 2 fine-tuning:  {self.results['finetune_epochs']} epochs, {t:.1f} min"
            )

        lines.append(f"\n{'─'*W}")
        lines.append("NOTES")
        lines.append(f"{'─'*W}")
        lines.append("  Layer benchmarks: single attention layer, d_model=768.")
        lines.append("  Full model: all layers replaced, end-to-end measurement.")
        lines.append("  PPL: WikiText-2 validation, block_size=256.")
        lines.append("  ZRO v0.2 — github.com/zrointelligence")
        lines.append("=" * W)
        return "\n".join(lines)

    def print_summary(self):
        print(self._build_report())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="ZRO Compiler v0.2: Replace transformer attention with O(n) linear attention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python zro_compile_v2.py --model gpt2 --mode benchmark_only
  python zro_compile_v2.py --model gpt2 --mode full --hedgehog
        """,
    )
    p.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    p.add_argument("--mode", default="full", choices=["benchmark_only", "distill_only", "full"])
    p.add_argument("--hedgehog", action="store_true", help="Use Hedgehog feature map")
    p.add_argument("--freeze_non_hedgehog", action="store_true", help="During distill, only train W1")
    p.add_argument("--distill_epochs", type=int, default=3)
    p.add_argument("--finetune_epochs", type=int, default=2)
    p.add_argument("--distill_lr", type=float, default=3e-5)
    p.add_argument("--finetune_lr", type=float, default=5e-6)
    p.add_argument("--chunk_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--no_pretrained_init", action="store_true")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("-f", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. CPU will be very slow.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output or f"zro_v2_report_{args.model}_{ts}.txt"

    compiler = ZROCompiler(
        model_name=args.model,
        chunk_size=args.chunk_size,
        verbose=True,
        use_hedgehog=args.hedgehog,
    )

    compiler.load()
    compiler.compile(pretrained_init=not args.no_pretrained_init)

    if args.mode == "benchmark_only":
        compiler.benchmark_layer(label="")
        compiler.benchmark_full_model(label="")
        compiler.print_summary()

    elif args.mode == "distill_only":
        compiler.benchmark_layer(label="before")
        compiler.benchmark_full_model(label="before")
        compiler.train(
            distill_epochs=args.distill_epochs,
            finetune_epochs=0,
            distill_lr=args.distill_lr,
            batch_size=args.batch_size,
            log_every=args.log_every,
            freeze_non_hedgehog=args.freeze_non_hedgehog,
        )
        compiler.benchmark_layer(label="after")
        compiler.benchmark_full_model(label="after")

    elif args.mode == "full":
        compiler.benchmark_layer(label="before")
        compiler.benchmark_full_model(label="before")
        compiler.train(
            distill_epochs=args.distill_epochs,
            finetune_epochs=args.finetune_epochs,
            distill_lr=args.distill_lr,
            finetune_lr=args.finetune_lr,
            batch_size=args.batch_size,
            log_every=args.log_every,
            freeze_non_hedgehog=args.freeze_non_hedgehog,
        )
        compiler.benchmark_layer(label="after")
        compiler.benchmark_full_model(label="after")

    compiler.save_report(report_path)
    return compiler


if __name__ == "__main__":
    main()
