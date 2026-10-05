# RDNA4 reevaluation — October 5, 2026

Physical Radeon RX 9070 (`gfx1201`) revalidation of all 21 external workloads in both modes at the September 30 ledger settings, including CLIP SuperCollider’s read-only delays. Of 45 clean configurations, 42 passed, TP1 prefill Default was rejected for incomplete runtime synchronization evidence, and two scatter-reduce configurations confirmed the existing global-memory scope limitation. Fractions count detections in eight admitted/reached fault trials; green requires at least 6/8 plus matching clean qualification, healthy GPU checks, and complete instrumentation and mutation evidence.

Measured source: `952b36716c2` on `shared/rocjitsu/sanitizers`, freshly rebuilt with GCC 15 (`-O2 -g`, assertions enabled). Fresh native discovery, exact-name allowlists, clean results, 312 fault trials, provenance, and qualification audits are retained in the [campaign directory](/home/benoit/workspace/consan-validation/rdna4-reevaluation-20261005). The September 30 ASan/UBSan report is historical; those suites were not rerun for this campaign.

The **Set** and **Priority** columns follow the validation manifest; **Workload / validation ID** identifies each external workload and its exact manifest ID. **Default** reports ConSan's default analysis mode, with the sampling preset selected separately for each row; **SuperCollider** reports the alternative detector that perturbs memory-access timing to expose races.

In Default cells, `high` and `higher` are values of `RJ_CONSAN_PRESET`, not separate detectors: `default` uses workgroup/LDS-cell sampling strides of 256/256, `high` uses 16/16, and `higher` uses 1/4. Smaller strides sample more densely and generally cost more. A bank count overrides the retained-access capacity. In SuperCollider cells, `sleep`, `sleep_wave`, and `nop` identify the timing-perturbation controls; a delay matrix uses the listed settings across the trial batch. See [presets](../USAGE.md#presets) and [expert controls](../EXPERT_CONTROLS.md) for details.

Global-only scatter-reduce remains outside LDS/FLAT coverage; see [the global-memory support analysis](../SUPERCOLLIDER_GLOBAL_MEMORY.md).

Shared [color scale](VALIDATION.md#status-colors): 🟩 qualified; 🟨 clean
but below the fault bar or outside scope; 🟧 incomplete qualification;
🟥 correctness/instrumentation failure; 🩶 pending.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| RDNA4 corpus | P0 | production FP16 matmul (`rdna4-matmul-fp16-production`) | 🟩 clean pass; fault 8/8; high | 🟩 clean pass; fault 8/8; sleep_wave=15 |
| RDNA4 corpus | P0 | production FP8 matmul (`rdna4-matmul-fp8-production`) | 🟩 clean pass; fault 8/8; high | 🟩 clean pass; fault 7/8; sleep_wave=15 |
| Main E2E | P0 | Qwen prefill (`qwen-prefill`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; sleep=15 |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟧 numerical pass; runtime synchronization evidence incomplete; fault not run; higher | 🟩 clean pass; fault 8/8; sleep_wave=15 |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; sleep=1 |
| PyTorch | P1 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟨 numerical pass; global-only scope limitation; default | 🟨 numerical pass; global-only scope limitation; nop=0 |
| PyTorch | P2 | compiled softmax (`pytorch-rdna4-compiled-softmax`) | 🟩 clean pass; fault 8/8; higher + 64 banks; same-value writes allowed | 🟩 clean pass; fault 6/8; sleep=1; same-value writes allowed |
| PyTorch | P2 | split softmax (`pytorch-rdna4-split-softmax`) | 🟩 clean pass; fault 8/8; high; same-value writes allowed | 🟩 clean pass; fault 8/8; nop delay matrix 0/16/64/256 |
| PyTorch | P2 | LLM top-k (`pytorch-rdna4-llm-topk`) | 🟨 clean pass; fault 2/8; high | 🟨 clean pass; fault 0/8; sleep=15 |
| llama.cpp | P2 | quantized matrix-vector multiply (`llama-rdna4-mul-mat-vec-q`) | 🟩 clean pass; fault 8/8; higher + 64 banks | 🟩 clean pass; fault 7/8; sleep=1 |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🟨 clean pass; fault 5/8; high | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟩 clean pass; fault 8/8; high | 🟩 clean pass; fault 8/8; sleep=2; reads-only delay |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟩 clean pass; fault 8/8; higher + 256 banks | 🟨 clean pass; fault 0/8; sleep=15 |
| llama.cpp | P3 | RMS normalization (`llama-rdna4-rms-norm`) | 🟩 clean pass; fault 8/8; higher | 🟩 clean pass; fault 8/8; sleep_wave=15 |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 clean pass; fault 8/8; high | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 clean pass; fault 8/8; high | 🟨 clean pass; fault 0/8; sleep=15 |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; nop=0 |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; nop=0 |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟩 clean pass; fault 8/8; higher | 🟨 clean pass; fault 0/8; sleep=15 |

TP1 prefill Default completed with numerical correctness and full static instrumentation (228/228 access sites and 60/60 barriers), but the runtime report contained `unsupported_sync=1` and the final verdict reported `dynamic_incomplete=1`. The initial run did not establish clean qualification, so the cell is orange and its fault batch was not run. Subsequent controls in current/September-30/September-30/current order all passed with identical higher-preset settings and fresh native comparators. These four runs did not reproduce the incomplete-sync report or establish a hook-version effect. The initial failure remains an unresolved intermittent qualification issue; passing repeats do not replace it. See the [matched controls](/home/benoit/workspace/consan-validation/rdna4-reevaluation-20261005/tp1-controls/comparison.json).

Compiled and split softmax required prospective fault-identity refreshes before any mutation: the freshly loaded ELF images differ from the September captures only in debug sections. Executable `.text`, all non-debug sections, kernel names, selected PCs, associated waits, and the reviewed reach proof are unchanged. The exact section comparison and refreshed specifications are retained under `softmax-identity-review/` and `softmax-identity-refresh/` in the campaign directory. Their matching clean runs use the same current inputs and controls.

Compared with September 30, TP2 Default falls from 6/8 to 5/8 and becomes yellow; compiled-softmax SuperCollider rises from 5/8 to 6/8 and becomes green. Top-k Default stays yellow at 2/8 (previously 5/8), and TP1 decode SuperCollider stays yellow at 0/8 (previously 3/8). These small batches do not establish source-version effects.

The table retains the first complete batch at the recorded settings; detection-rate fluctuations do not establish a source-version effect. All qualified cells require fresh matching clean controls and audited admission, reach, mutation installation, unchanged input hashes, complete runtime evidence, and healthy pre/post GPU checks. Eight-trial observed rates are not confidence bounds or proof of exhaustive race detection. The previous table is retained as `STATUS_RDNA4.baseline.md` in the campaign directory; the [September 30 regression review](REEVALUATION_RDNA4_20260930.md) remains historical evidence.

Benchmark results are separate from fault qualification; see the [gfx1201 benchmark ledger](../benchmark/STATUS_GFX1201.md).
