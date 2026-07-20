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
======================================================================
ZRO INTELLIGENCE — COMPILATION REPORT v0.2
Generated: 2026-07-19 12:35:34
======================================================================

MODEL:   gpt2
HEDGEHOG: False
DEVICE:  cuda
CHUNK:   256 tokens
ZRO PARAMS: 35,435,520

──────────────────────────────────────────────────────────────────────
LANGUAGE MODEL QUALITY (WikiText-2 Validation PPL)
──────────────────────────────────────────────────────────────────────
  Original gpt2: 44.46
  After Stage 1 distillation: 89.86  (gap +102.1%)
  After Stage 2 fine-tuning:  47.44  (gap +6.7%)

──────────────────────────────────────────────────────────────────────
LAYER-LEVEL MEMORY BENCHMARK (BEFORE)
──────────────────────────────────────────────────────────────────────
    Tokens |     Attn MB     ZRO MB   Saved |   Attn tok/s   ZRO tok/s
  ──────────────────────────────────────────────────────────────────
       512 |      707.6      992.2   -40.2% |      232,701      70,289
      1024 |      792.2     1000.6   -26.3% |      178,940      71,202
      2048 |     1116.3     1022.6     8.4% |      137,101      89,062
      4096 |     2374.0     1066.1    55.1% |       86,863      92,865
      8192 |     7331.7     1154.2    84.3% |       41,781      92,401

──────────────────────────────────────────────────────────────────────
LAYER-LEVEL MEMORY BENCHMARK (AFTER)
──────────────────────────────────────────────────────────────────────
    Tokens |     Attn MB     ZRO MB   Saved |   Attn tok/s   ZRO tok/s
  ──────────────────────────────────────────────────────────────────
       512 |     1405.1     1690.3   -20.3% |      369,447      91,575
      1024 |     1489.8     1698.2   -14.0% |      287,166      89,107
      2048 |     1813.8     1720.2     5.2% |      151,602      89,048
      4096 |     3073.2     1764.2    42.6% |       81,114      89,370
      8192 |     8030.6     1852.3    76.9% |       39,150      89,270

──────────────────────────────────────────────────────────────────────
FULL MODEL MEMORY BENCHMARK (BEFORE)
──────────────────────────────────────────────────────────────────────
    Tokens |     Orig MB     ZRO MB   Saved |   Orig tok/s   ZRO tok/s
  ──────────────────────────────────────────────────────────────────
       128 |     1681.9     1754.3    -4.3% |       12,362       4,210
       256 |     1718.4     1862.0    -8.4% |       14,280       5,251
       512 |     1801.7     1971.6    -9.4% |       13,309       5,323
      1024 |     1930.8     1981.6    -2.6% |       12,715       5,327

──────────────────────────────────────────────────────────────────────
FULL MODEL MEMORY BENCHMARK (AFTER)
──────────────────────────────────────────────────────────────────────
    Tokens |     Orig MB     ZRO MB   Saved |   Orig tok/s   ZRO tok/s
  ──────────────────────────────────────────────────────────────────
       128 |     2385.8     2458.2    -3.0% |       11,450       4,633
       256 |     2421.3     2565.9    -6.0% |       12,522       5,051
       512 |     2504.3     2673.4    -6.7% |       12,328       5,035
      1024 |     2635.6     2685.5    -1.9% |       11,813       5,053

──────────────────────────────────────────────────────────────────────
TRAINING SUMMARY
──────────────────────────────────────────────────────────────────────
  Stage 1 distillation: 3 epochs, 66.6 min
  Stage 2 fine-tuning:  2 epochs, 50.9 min

──────────────────────────────────────────────────────────────────────
NOTES
──────────────────────────────────────────────────────────────────────
  Layer benchmarks: single attention layer, d_model=768.
  Full model: all layers replaced, end-to-end measurement.
  PPL: WikiText-2 validation, block_size=256.
  ZRO v0.2 — github.com/zrointelligence
======================================================================

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
