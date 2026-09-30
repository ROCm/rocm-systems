# RDNA4 reevaluation — September 30, 2026

Physical gfx1201 validation of 21 external workloads in both modes, initially using the settings qualified in the September 26 ledger. All 43 applicable clean configurations passed. CLIP SuperCollider was subsequently requalified with read-only delays, as explained below. Two additional scatter-reduce checks confirmed the existing global-memory scope limitation. Fractions count detections in eight admitted/reached fault trials; green requires at least 6/8, matching clean qualification, healthy GPU checks, and complete instrumentation and mutation evidence.

Measured source: `72a843d3b61` on `shared/rocjitsu/sanitizers`. Fresh discovery, allowlists, clean results, 320 fault trials, provenance, and qualification audits are retained in the [campaign directory](/home/benoit/workspace/consan-validation/rdna4-reevaluation-20260930). The [Clang sanitizer report](CLANG_SANITIZERS_20260930.md) covers the complete ASan and UBSan suites on the same executable sources.

The **Set** and **Priority** columns follow the validation manifest; **Workload / validation ID** identifies each external workload and its exact manifest ID. **Default** reports ConSan's default analysis mode, with the sampling preset selected separately for each row; **SuperCollider** reports the alternative detector that perturbs memory-access timing to expose races.

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
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 3/8; sleep=1 |
| PyTorch | P1 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟨 numerical pass; global-only scope limitation; default | 🟨 numerical pass; global-only scope limitation; nop=0 |
| PyTorch | P2 | compiled softmax (`pytorch-rdna4-compiled-softmax`) | 🟩 clean pass; fault 8/8; higher + 64 banks; same-value writes allowed | 🟨 clean pass; fault 5/8; sleep=1; same-value writes allowed |
| PyTorch | P2 | split softmax (`pytorch-rdna4-split-softmax`) | 🟩 clean pass; fault 8/8; high; same-value writes allowed | 🟩 clean pass; fault 8/8; nop delay matrix 0/16/64/256 |
| PyTorch | P2 | LLM top-k (`pytorch-rdna4-llm-topk`) | 🟨 clean pass; fault 5/8; high | 🟨 clean pass; fault 0/8; sleep=15 |
| llama.cpp | P2 | quantized matrix-vector multiply (`llama-rdna4-mul-mat-vec-q`) | 🟩 clean pass; fault 8/8; higher + 64 banks | 🟩 clean pass; fault 8/8; sleep=1 |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🟩 clean pass; fault 6/8; high | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟩 clean pass; fault 8/8; high | 🟩 clean pass; fault 8/8; sleep=2; reads-only delay |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟩 clean pass; fault 8/8; higher + 256 banks | 🟨 clean pass; fault 0/8; sleep=15 |
| llama.cpp | P3 | RMS normalization (`llama-rdna4-rms-norm`) | 🟩 clean pass; fault 8/8; higher | 🟩 clean pass; fault 8/8; sleep_wave=15 |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 clean pass; fault 8/8; high | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 clean pass; fault 8/8; high | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; nop=0 |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; nop=0 |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; sleep=15 |

Top-k Default remains yellow at the initial 5/8. Matched controls with unchanged high-preset settings detected 6/8 with both the September 26 and current hooks; all control trials had matching clean qualification and complete evidence. That small comparison did not reproduce a hook-version effect and does not prove equivalence. The passing repeat does not replace the initial result. Across the two current batches, 11/16 faults were detected. See the [regression review](REEVALUATION_RDNA4_20260930.md) and [matched controls](/home/benoit/workspace/consan-validation/rdna4-reevaluation-20260930/topk-controls/comparison.json).

Compiled-softmax SuperCollider remains yellow at its initial 5/8 (previously 8/8). Matched controls detected 6/8 with the old hook and 7/8 with the current hook. WMMA attention fell from 5/8 to 0/8; both matched controls detected 0/8. These small comparisons did not reproduce a hook-version effect. Their initial results remain in the table.

CLIP SuperCollider initially fell from 8/8 to 2/8 with sleep=2 applied to reads and writes. Matched controls detected 5/8 old versus 1/8 current; reversing the order gave 3/8 current versus 5/8 old. Increasing the delay alone did not qualify: sleep=4 detected 5/8 and sleep=3 detected 2/8. Applying sleep=2 only to reads restored qualification in two independent batches of 8/8, each with fresh matching clean/native checks and inventory. The table selects the confirmation batch and explicitly labels this changed control. This is detector calibration, not a source-code fix or proof of the initial drop’s cause.

For compiled-softmax, CLIP, and WMMA, all historical/current patched images at the original settings are byte-identical after normalizing only their logged report-buffer addresses. SuperCollider’s host report registry is also unchanged. Timing sensitivity remains relevant: green means the recorded configuration met this campaign’s statistical gate, not exhaustive race detection. The [regression review](REEVALUATION_RDNA4_20260930.md) preserves all initial results and controls. In total, this reevaluation includes 320 initial fault trials and 112 follow-up trials.
