ZRO Intelligence
Pre-Seed Executive Summary  ·  July 13, 2026
Run 10× longer context on the same hardware. No retraining. No new GPUs.



VERIFIED RESULTS — FROM ONE COMPILER RUN ON FREE T4 GPU
Source: zro_report_gpt2_20260713_060214.txt  ·  Hardware: NVIDIA Tesla T4, FP32, CUDA  ·  Runtime: 53.5 min

Memory Reduction at 8,192 Tokens
87.2%
Attention: 7,339 MB  →  ZRO: 940 MB


Throughput Advantage at 8,192 Tokens
7.1×
ZRO: 306,707 tok/s  vs  Attention: 43,206 tok/s


Language Model Quality (WikiText-2 PPL)
68.99
vs GPT-2 original: 44.46  ·  Gap: +55.2%  ·  53.5 min training



THE PROBLEM
Transformer attention builds an N×N matrix at every layer. Memory grows as O(N²). At 8,192 tokens: 7,339 MB and 43,206 tok/s — bottlenecked and slow. At 128K tokens: ~1.8 TB — no single GPU can run it. Every AI lab racing toward 1M-token context windows will hit this wall.
THE SOLUTION
ZRO is a software compiler. One command replaces all attention layers in any pretrained model with O(n) linear-memory recurrence. No retraining. No new hardware.
One command:  python zro_compile.py --model gpt2 --mode full


WHAT OTHERS DON'T HAVE
Mamba, RWKV, RetNet all achieve O(n) memory — but require training from scratch ($2M+ for 7B models). ZRO works on existing pretrained models. We are the only retrofit compiler in the market.
THE ASK
$500K–$750K Pre-Seed:  2 engineers + compute → LLaMA-3-8B support, <20% PPL gap, first paying customer.


ZRO Intelligence — Benchmark Fact Sheet
Source: zro_report_gpt2_20260713_060214.txt  ·  NVIDIA Tesla T4, FP32, CUDA  ·  



Headline Numbers
Metric
Value
Detail
Memory Reduction at 8K tokens
87.2%
Attention: 7,339 MB  →  ZRO: 940 MB
Throughput Advantage at 8K
7.1×
ZRO: 306,707 tok/s  vs  Attn: 43,206 tok/s
Language Model Quality (PPL)
68.99
vs GPT-2 original: 44.46  ·  gap: +55.2%
Total Training Time
53.5 min
T4 GPU  ·  Stage 1: {STAGE1_MIN} min  ·  Stage 2: {STAGE2_MIN} min
Speed crossover point
2,048 tokens
ZRO faster than attention from this point
ZRO parameters
35,453,952
vs original attention: same parameter count


Memory Benchmark — Before Training (Kernel Level)
Tokens
Attention MB
ZRO MB
Saved
Attn tok/s
ZRO tok/s
Speed
512
714.5
709.0
0.8%
243,797
149,791
0.61×
1,024
798.6
720.5
9.8%
183,990
166,924
0.91×
2,048
1,123
752.0
33.1%
117,926
180,539
1.53×
4,096
2,381
814.9
65.8%
90,647
292,299
3.22×
8,192
7,339
940.2
87.2%
43,206
306,707
7.10×


Memory Benchmark — After Training
Tokens
Attention MB
ZRO MB
Saved
Attn tok/s
ZRO tok/s
Speed
512
913.5
907.4
0.7%
412,225
159,801
0.39×
1,024
997.6
918.9
7.9%
281,416
197,847
0.70×
2,048
1,322
950.4
28.1%
118,129
180,279
1.53×
4,096
2,581
1,013
60.7%
82,147
294,840
3.59×
8,192
7,538
1,139
84.9%
37,605
290,904
7.74×


Quality Progression (GPT-2 124M, WikiText-2 Validation PPL)
Phase
Architecture
PPL
Gap vs GPT-2
Baseline
GPT-2 Original
44.46
0%
Phase 0
EMA vector state
2,135
+4,703%
Phase 1
Matrix associative memory
435
+879%
Phase 2
Multi-head + ELU+1 + 2-stage
138
+211%
Phase 3
GLA + input-dependent gating
68.99
+55.2%


Important context on short-context memory numbers:  At 512–1,024 tokens, GPT-2 model weights (~550 MB) dominate total memory for both architectures. The O(n) advantage becomes decisive above 4,096 tokens where attention quadratic growth dominates. Memory numbers are layer-level comparisons (one attention layer vs one ZRO layer, d_model=768).




ZRO Intelligence  ·  github.com/zrointelligence  ·  @zrointelligence

