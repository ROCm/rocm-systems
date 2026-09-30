# RDNA4 reevaluation — September 26, 2026

Physical gfx1201 validation of the external workloads. Initial runs use recorded settings. Added-bank settings and CLIP sleep=2 are requalifications; the [regression review](/home/benoit/workspace/consan-validation/rdna4-reevaluation-20260926/REGRESSIONS.md) preserves the initial failures. Fractions count detections in eight admitted/reached fault trials; green requires at least 6/8 and matching clean qualification. Pending cells do not inherit historical grades. Raw evidence stays in the [external campaign directory](/home/benoit/workspace/consan-validation/rdna4-reevaluation-20260926).

The **Workload** column identifies the external workload. **Default** reports ConSan's default analysis mode, with the sampling preset selected separately for each row; **SuperCollider** reports the alternative detector that perturbs memory-access timing to expose races.

In Default cells, `high` and `higher` are values of `RJ_CONSAN_PRESET`, not separate detectors: `default` uses workgroup/LDS-cell sampling strides of 256/256, `high` uses 16/16, and `higher` uses 1/4. Smaller strides sample more densely and generally cost more. A bank count overrides the retained-access capacity. In SuperCollider cells, `sleep`, `sleep_wave`, and `nop` identify the timing-perturbation controls; a delay matrix uses the listed settings across the trial batch. See [presets](../USAGE.md#presets) and [expert controls](../EXPERT_CONTROLS.md) for details.

Global-only scatter-reduce remains outside LDS/FLAT coverage; see [the global-memory support analysis](../SUPERCOLLIDER_GLOBAL_MEMORY.md).

Shared [color scale](VALIDATION.md#status-colors): 🟩 qualified; 🟨 clean
but below the fault bar or outside scope; 🟧 incomplete qualification;
🟥 correctness/instrumentation failure; 🩶 pending.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| RDNA4 corpus | P0 | production FP16 matmul (`rdna4-matmul-fp16-production`) | 🟩 clean pass; fault 8/8; high | 🟩 clean pass; fault 7/8; sleep_wave=15 |
| RDNA4 corpus | P0 | production FP8 matmul (`rdna4-matmul-fp8-production`) | 🟩 clean pass; fault 8/8; high | 🟩 clean pass; fault 7/8; sleep_wave=15 |
| Main E2E | P0 | Qwen prefill (`qwen-prefill`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; sleep=15 |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟩 clean pass; fault 8/8; higher | 🟩 clean pass; fault 8/8; sleep_wave=15 |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 1/8; sleep=1 |
| PyTorch | P1 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟨 numerical pass; global-only scope limitation; default | 🟨 numerical pass; global-only scope limitation; nop=0 |
| PyTorch | P2 | compiled softmax (`pytorch-rdna4-compiled-softmax`) | 🟩 clean pass; fault 8/8; higher + 64 banks; same-value writes allowed | 🟩 clean pass; fault 8/8; sleep=1; same-value writes allowed |
| PyTorch | P2 | split softmax (`pytorch-rdna4-split-softmax`) | 🟩 clean pass; fault 8/8; high; same-value writes allowed | 🟩 clean pass; fault 7/8; nop delay matrix 0/16/64/256 |
| PyTorch | P2 | LLM top-k (`pytorch-rdna4-llm-topk`) | 🟩 clean pass; fault 7/8; high | 🟨 clean pass; fault 0/8; sleep=15 |
| llama.cpp | P2 | quantized matrix-vector multiply (`llama-rdna4-mul-mat-vec-q`) | 🟩 clean pass; fault 8/8; higher + 64 banks | 🟩 clean pass; fault 8/8; sleep=1 |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🟩 clean pass; fault 7/8; high | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟩 clean pass; fault 8/8; high | 🟩 clean pass; fault 8/8; sleep=2 |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟩 clean pass; fault 7/8; higher + 256 banks | 🟨 clean pass; fault 0/8; sleep=15 |
| llama.cpp | P3 | RMS normalization (`llama-rdna4-rms-norm`) | 🟩 clean pass; fault 8/8; higher | 🟩 clean pass; fault 8/8; sleep_wave=15 |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 clean pass; fault 8/8; high | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 clean pass; fault 8/8; high | 🟨 clean pass; fault 5/8; sleep=15 |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; nop=0 |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; nop=0 |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; sleep=15 |
