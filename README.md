# ZRO Intelligence — Transformer Attention Compiler

Replace quadratic attention with linear attention via distillation. No retraining. No new data. One command.

`bash
python zro_compile.py --model gpt2 --mode full

What this is
Every transformer uses attention, which builds an N×N matrix at every layer. Memory grows as O(N²). At 128K tokens, this is ~1.8 TB — physically impossible on one GPU.
ZRO is a compiler that replaces this attention mechanism with a linear-memory recurrent alternative through distillation. You give it a pretrained model. It returns a compiled model that uses constant memory in sequence length.

Metric                                         |    Result,
Perplexity gap                                 |    +6.7% (44.46 → 47.44),
Training time                                  |    117 min on NVIDIA T4 (free Colab GPU),
Layer-level memory at 8K                       |    84% reduction,
Method                                         |      Distillation + fine-tuning, no training data required,

[zro_v2_report_gpt2_20260719_103541.txt](https://github.com/user-attachments/files/30176407/zro_v2_report_gpt2_20260719_103541.txt)

Architecture: GLA-CMLA
 
Gated Linear Attention with per-dimension gating (not scalar decay)
 
Causal Multi-Head structure matching GPT-2 exactly
 
ELU+1 feature map with optional Hedgehog learned map
 
Chunk-wise parallel recurrence via cumulative products
 
Pretrained weight initialization — starts from model's own Q/K/V projections


Quick Start
Colab (Recommended — zero setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o6uQmAK5-MPgakmalBaQuH7hXzfrk_v_?usp=sharing)

Local
pip install torch transformers datasets
python zro_compile_v2.py --model gpt2 --mode full

How it works
Original Model
     ↓
[ZRO Compiler]
  ├─ Auto-detect attention layers
  ├─ Replace with GLA-CMLA module
  ├─ Initialize from pretrained Q/K/V weights
  ├─ Stage 1: Layer-by-layer distillation (mimic teacher outputs)
  ├─ Stage 2: Language model fine-tuning (recover fluency)
  └─ Benchmark before/after
     ↓
Compiled Model

Honest Limitations
We are pre-seed and this is v0.2. Here is what we know and don't know:

Claim                                                                 |       Status 
O(n) memory scaling proven                                            |      ✅ Yes — at layer level, 84% at 8K tokens 
Quality preserved                                                     |      ✅ Yes — +6.7% PPL gap on GPT-2 
Works on modern models (Llama-3)                                      |      🔄 Not yet — 6-week roadmap 
Full-model speedup                                                    |      ❌ No — PyTorch loops are 2.4× slower than Flash Attention. Triton kernel required. 
Short-context savings (<2K)                                           |      ❌ No — model weights dominate. Advantage starts at 4K+. 
Task benchmarks (MMLU)                                                |      ⚠️ GPT-2 scores near random on MMLU. Need Llama-3 for meaningful test.

Why this matters
Google, Meta, and OpenAI are building new architectures (Mamba, RWKV, RetNet). If you trained a model for $10M, you must retrain from scratch to use them.
ZRO is the only approach that retrofits existing pretrained weights to linear attention. The compiler is the product.
Roadmap
 
GPT-2 compiler with distillation pipeline
 
Per-dimension gating (true GLA)
 
Llama-3-8B port
 
Triton kernel for production speed
 
MMLU / HellaSwag task benchmarks on modern models
 
vLLM / TensorRT-LLM serving integration

Citation
@software{zro_intelligence_2026,
  author = {Gajare, Mayank},
  title = {ZRO Intelligence: A Compiler for Linear Attention Distillation},
  year = {2026},
  url = {https://github.com/zrointelligence}
}

License
Apache 2.0
