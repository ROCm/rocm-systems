# ConSan RDNA4 (`gfx1201`) status

September 23, 2026 revalidation is in progress. The table records fresh
assessments; the campaign log retains prior attempts. Results qualify only
their recorded binaries and configurations. Rerun the linked procedure after
relevant changes.

Start any new revalidation with
[rocprofv3-based allowlist discovery and application](VALIDATION.md#first-step-for-revalidation-generate-and-apply-kernel-allowlists).
Apply this before retrying recorded timeouts or raising deadlines; update their
status only after new runs provide evidence.

This ledger summarizes accepted physical `gfx1201` evidence. Green rows have
the clean, coverage, reviewed-fault, containment, health, and
provenance evidence required by their row contract. Exact counts belong to the
binaries used for qualification and must be refreshed after relevant source,
toolchain, workload, or runtime changes.

## September 23 revalidation in progress

All 42 workload/mode cells are being rerun, including previously green cells.
Every cell has a fresh clean assessment. Reviewed fault qualification remains
incomplete; historical green results do not count as passes for this campaign.

- Host: AMD Radeon RX 9070 (`gfx1201`); native HIP shared-memory smoke passed.
- SDK: `/home/benoit/venv`, ROCm `10.2.0a20260915` development package.
- Preparation: current hook and production matmul built; generated per-workload
  allowlists applied; Qwen build provenance verified through the campaign symlinks.
- Fresh clean assessments: **42/42**, with **40 clean passes** across recorded
  configurations. No cell has passed fresh fault qualification.
- Preparation logs: `/home/benoit/workspace/consan-validation-artifacts/`.

Legend: 🩶 unseen · 🟥 broken before useful evidence · 🟧 below 80% aggregate applicable-site support or another substantial gap · 🟨 timeout-only blocker or at least 80% aggregate applicable-site support · 🟩 accepted workload/profile contract.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Production HIP | P0 | FP16 matmul (`rdna4-matmul-fp16-production`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Production HIP | P0 | FP8 matmul (`rdna4-matmul-fp8-production`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending; prior fault assessment: standard fault 0/8; preset=max clean passes and fault detects 7/8 (separate configuration) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending; prior fault assessment: clean pass; reviewed publication-barrier fault missed in 8/8 trials |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending; prior fault assessment: standard fault 0/8; preset=max clean passes and fault detects 8/8 (separate configuration) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending; prior fault assessment: clean pass; reviewed cross-wave publication fault missed in 8/8 trials |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P1 | collision-heavy `scatter_reduce` (`pytorch-scatter-reduce`) | 🟥 Sep 23: no applicable code object; analysis incomplete | 🟥 Sep 23: no applicable code object; analysis incomplete |
| PyTorch | P2 | Inductor compiled softmax (`pytorch-rdna4-compiled-softmax`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P2 | split online softmax (`pytorch-rdna4-split-softmax`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P2 | Qwen-vocabulary top-k (`pytorch-rdna4-llm-topk`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| llama.cpp | P2 | quantized matvec (`llama-rdna4-mul-mat-vec-q`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P3 | native histogram (`pytorch-torch-histc`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| llama.cpp | P3 | RMS norm (`llama-rdna4-rms-norm`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟨 Sep 23: clean pass; grouped publication fault 0/8 detections; fault qualification failed | 🟨 Sep 23: clean pass; grouped publication fault 0/8 detections; fault qualification failed |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |

### September 23 cell evidence

- `d128-block` / Default, attempt 01: rocprofv3 discovered five exact kernel
  names; native baseline passed. The current hook rejects the transformed image
  because indirect call target recovery is unavailable during text relocation.
  Exit 92; no clean or fault qualification claimed. Evidence:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/attempt-01/d128-block/`.
- Host ConSan gate: 1,055 tests passed in the fresh build. Validation runner:
  191 tests passed, including generated-allowlist environment tests.
- `d128-block` / SuperCollider, attempt 02: same strict relocation rejection,
  exit 92. Evidence:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/attempt-02/d128-block/`.
- Qwen preparation and production matmul compilation now pass. All 245 hook
  unit tests passed; the labeled nonphysical gate completed (see preparation log).
- `pytorch-torch-mode` / default: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-torch-mode-default`.
- `pytorch-torch-mode` / supercollider: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-torch-mode-supercollider`.
- `pytorch-scatter-reduce` / default: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-scatter-reduce-default`.
- `pytorch-scatter-reduce` / supercollider: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-scatter-reduce-supercollider`.
- `pytorch-torch-histc` / default: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-torch-histc-default`.
- `pytorch-torch-histc` / supercollider: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-torch-histc-supercollider`.
- `pytorch-rdna4-compiled-softmax` / default: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-rdna4-compiled-softmax-default`.
- `pytorch-rdna4-compiled-softmax` / supercollider: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-rdna4-compiled-softmax-supercollider`.
- `pytorch-rdna4-split-softmax` / default: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-rdna4-split-softmax-default`.
- `pytorch-rdna4-split-softmax` / supercollider: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-rdna4-split-softmax-supercollider`.
- `pytorch-rdna4-llm-topk` / default: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-rdna4-llm-topk-default`.
- `pytorch-rdna4-llm-topk` / supercollider: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round1/pytorch-rdna4-llm-topk-supercollider`.
- PyTorch discovery repair: native validation now exits normally so rocprofv3
  can flush its CSV. A fresh `torch.mode` native profile passed its oracle and
  produced two exact kernel names. PyTorch uses the existing
  `/home/benoit/gpu-venv-r102` stack (`2.14.0+rocm10.2.0a20260918`) and its
  matching profiler; the older PyTorch venv loaded incompatible profiler SDKs.
  The initial discovery failures above are being retried with this fix.
- All 2,013 labeled nonphysical tests and all 389 gfx1201 physical fixture tests
  passed. These gates do not substitute for the external workload cells.
- D128 Default attempt 03, rebuilt with the September 15 SDK: native pass,
  same indirect-call relocation failure (now `.text+59912`). Compiler refresh
  alone does not resolve this failure.
- `pytorch-torch-mode` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-torch-mode-default`.
- `pytorch-torch-mode` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-torch-mode-supercollider`.
- `pytorch-scatter-reduce` / default: no applicable code object; analysis incomplete. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-scatter-reduce-default`.
- `pytorch-scatter-reduce` / supercollider: no applicable code object; analysis incomplete. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-scatter-reduce-supercollider`.
- `pytorch-torch-histc` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-torch-histc-default`.
- `pytorch-torch-histc` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-torch-histc-supercollider`.
- `pytorch-rdna4-compiled-softmax` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-rdna4-compiled-softmax-default`.
- `pytorch-rdna4-compiled-softmax` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-rdna4-compiled-softmax-supercollider`.
- `pytorch-rdna4-split-softmax` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-rdna4-split-softmax-default`.
- `pytorch-rdna4-split-softmax` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-rdna4-split-softmax-supercollider`.
- `pytorch-rdna4-llm-topk` / default: no applicable code object; analysis incomplete. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-rdna4-llm-topk-default`.
- `pytorch-rdna4-llm-topk` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/pytorch-rdna4-llm-topk-supercollider`.
- `rdna4-matmul-fp16-production` / default: missing ConSan coverage record; exit 124. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/rdna4-matmul-fp16-production-default`.
- `rdna4-matmul-fp16-production` / supercollider: missing ConSan coverage record; exit 124. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/rdna4-matmul-fp16-production-supercollider`.
- `rdna4-matmul-fp8-production` / default: missing ConSan analysis verdict; exit 124. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/rdna4-matmul-fp8-production-default`.
- `rdna4-matmul-fp8-production` / supercollider: missing ConSan analysis verdict; exit 124. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/rdna4-matmul-fp8-production-supercollider`.
- `qwen-prefill` / default: exit 1. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/qwen-prefill-default`.
- `qwen-prefill` / supercollider: exit 1. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/qwen-prefill-supercollider`.
- `tp1-prefill` / default: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/tp1-prefill-default`.
- `tp1-prefill` / supercollider: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/tp1-prefill-supercollider`.
- `tp1-decode-combined` / default: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/tp1-decode-combined-default`.
- `tp1-decode-combined` / supercollider: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/tp1-decode-combined-supercollider`.
- `tp2-family` / default: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/tp2-family-default`.
- `tp2-family` / supercollider: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/tp2-family-supercollider`.
- `clip-bf16` / default: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/clip-bf16-default`.
- `clip-bf16` / supercollider: native rocprofv3 discovery failed; inspect discovery log. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/clip-bf16-supercollider`.
- `llama-rdna4-mul-mat-vec-q` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/llama-rdna4-mul-mat-vec-q-default`.
- `llama-rdna4-mul-mat-vec-q` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/llama-rdna4-mul-mat-vec-q-supercollider`.
- `llama-rdna4-rms-norm` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/llama-rdna4-rms-norm-default`.
- `llama-rdna4-rms-norm` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/llama-rdna4-rms-norm-supercollider`.
- `d128-pressure` / default: ConSan rejected a code object before execution; exit 92. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/d128-pressure-default`.
- `d128-pressure` / supercollider: ConSan rejected a code object before execution; exit 92. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/d128-pressure-supercollider`.
- `wmma-attention` / default: ConSan rejected a code object before execution; exit 92. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/wmma-attention-default`.
- `wmma-attention` / supercollider: ConSan rejected a code object before execution; exit 92. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/wmma-attention-supercollider`.
- `streamk-arrival` / default: no applicable code object; analysis incomplete. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/streamk-arrival-default`.
- `streamk-arrival` / supercollider: no applicable code object; analysis incomplete. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/streamk-arrival-supercollider`.
- `tree-atomic-or` / default: no applicable code object; analysis incomplete. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/tree-atomic-or-default`.
- `tree-atomic-or` / supercollider: no applicable code object; analysis incomplete. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/tree-atomic-or-supercollider`.
- `jakub-attention` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/jakub-attention-default`.
- `jakub-attention` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round2/jakub-attention-supercollider`.
- `tp1-prefill` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/tp1-prefill-default`.
- `tp1-prefill` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/tp1-prefill-supercollider`.

### September 23 follow-up fixes and diagnosis

- Out-of-tree access checked with `doctor --workload all`: all workload sources,
  executables, libraries, weights, and input/expected data remain accessible.
  The check exposed a Qwen manifest path mismatch through campaign symlinks;
  canonicalizing the encoder path fixed it without rebuilding or weakening hashes.
- Validation runner: 193 tests pass, including symlinked Qwen provenance,
  retaining matmul child diagnostics at timeout, and probing Sharktank imports
  in the selected interpreter. The repaired Sharktank runtime probe passes.
- Production matmul now honors `SKIP_BENCH` after both numerical checks. Native
  FP16 and FP8 profiling passes, with three exact kernel names each. External
  source fix: `sanitizer-strategy` local commit `16c5ad0`.
- FP8 Default's 60-second failure spent 58.3 seconds in inventory and reached
  patching before the deadline. A separate 300-second diagnostic retry is queued;
  any result uses a new artifact root and records the changed deadline.
- Qwen's native baseline and native profile pass. Both instrumented modes fail
  the numeric oracle at output index 151936 (4.23818 versus 5.36443) despite
  complete coverage reports. Coverage completeness does not qualify this cell.
- Sharktank's first discovery attempt lacked IREE Python bindings. The retry uses
  the local `iree-build` compiler/runtime bindings via recorded `PYTHONPATH` and
  `ml_dtypes` 0.6.0 in the existing SDK venv. TP1 prefill now passes both clean modes.
- An isolated Release hip-moi build completed successfully to investigate the
  unoptimized binaries' indirect-call and shared-helper limitations. A compiler
  optimization change requires new discovery, binary hashes, and fault review;
  it does not resolve the recorded unoptimized-binary failures by itself.
- `tp1-decode-combined` / default: missing ConSan analysis verdict; exit 124. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/tp1-decode-combined-default`.
- `tp1-decode-combined` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/tp1-decode-combined-supercollider`.
- `tp2-family` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/tp2-family-default`.
- `tp2-family` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/tp2-family-supercollider`.
- `clip-bf16` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/clip-bf16-default`.
- `clip-bf16` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/clip-bf16-supercollider`.
- `rdna4-matmul-fp16-production` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/rdna4-matmul-fp16-production-default`.
- `rdna4-matmul-fp16-production` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/rdna4-matmul-fp16-production-supercollider`.
- `rdna4-matmul-fp8-production` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/rdna4-matmul-fp8-production-default`.

- Qwen relocation diagnosis: the moved initializer retained its original
  `s_getpc_b64`-relative constant displacement. The data-address decoder missed
  RDNA4 `s_sext_i32_i16` canonicalization and `s_add_co_u32` /
  `s_add_co_ci_u32`. Local fix `1b9668bdbb3` recognizes that sequence and preserves
  carry across `s_wait_alu`; regression cases reject high-half and SCC clobbers.
  All 155 selected relocation/CFG host tests pass. An offline transform of the
  actual Qwen ELF now adjusts the constant displacement correctly. GPU retry
  remains pending; these host results do not promote either Qwen cell.
  Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/qwen-debug/`.
- Matmul follow-up: FP16 Default and SuperCollider, and FP8 Default, pass clean
  qualification with the explicit 300-second diagnostic deadline. FP8 Default
  spends about 58 seconds in preparation and another 58 seconds in patching;
  the original 60-second contract is still exceeded. Fault qualification remains
  pending. The table records the timeout override, and original failures remain
  in the earlier artifact roots.
- `rdna4-matmul-fp8-production` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round3/rdna4-matmul-fp8-production-supercollider`.
- `qwen-prefill` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/qwen-prefill-default`.
- `qwen-prefill` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/qwen-prefill-supercollider`.
- `d128-block` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/d128-block-default`.
- `d128-block` / supercollider: exit 1. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/d128-block-supercollider`.
- `d128-pressure` / default: exit 1. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/d128-pressure-default`.
- `d128-pressure` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/d128-pressure-supercollider`.
- `wmma-attention` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/wmma-attention-default`.
- `wmma-attention` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/wmma-attention-supercollider`.
- `streamk-arrival` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/streamk-arrival-default`.
- `streamk-arrival` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/streamk-arrival-supercollider`.
- `tree-atomic-or` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/tree-atomic-or-default`.
- `tree-atomic-or` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/tree-atomic-or-supercollider`.
- `jakub-attention` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/jakub-attention-default`.
- `jakub-attention` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round4/jakub-attention-supercollider`.
- Qwen Default reviewed fault: all eight trials installed exactly one paired
  publication-barrier drop, completed normally, passed the independent oracle,
  and passed pre/post GPU health probes. Detection was 0/8, below the precommitted
  minimum of one. This is a failed detection qualification, not a green cell.
  Reach is supported by the reviewed unconditional ISA path. Evidence:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/fault-round1-qwen-default/`.
- With the rebuilt relocation fix, Qwen passes both clean modes. All 1,055
  host ConSan tests pass again. Both matmul modes and precisions pass with the
  measured 300-second envelope, now adopted by the workload manifest; their
  earlier override runs remain explicitly recorded.
- Qwen SuperCollider reviewed fault: 0/8 detections with the precommitted
  delay settings (0, 4, 16, 64 NOPs, twice each). All mutations were installed,
  all numerical oracles passed, and all pre/post health checks passed. The
  minimum-detection requirement failed; the cell remains yellow. Evidence:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/fault-round1-qwen-supercollider/`.
- At the end of round 4, 36/42 clean assessments passed. Round 6 below
  resolves the three illegal-instruction failures; historical failures remain
  in their artifact roots.
- `tp1-decode-combined` / default: missing ConSan analysis verdict; exit 124. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round5/tp1-decode-combined-default`.
- `tp1-decode-combined` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round5/tp1-decode-combined-supercollider`.
- TP1 decode Default retry with the repaired hook reproduces `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`, followed by a 300-second teardown timeout. Its native baseline and SuperCollider run pass. A longer deadline does not resolve this failure. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round5/`.
- `d128-block` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round6/d128-block-default`.
- `d128-block` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round6/d128-block-supercollider`.
- `d128-pressure` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round6/d128-pressure-default`.
- `d128-pressure` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round6/d128-pressure-supercollider`.
- `tp1-decode-combined` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round6/tp1-decode-combined-default`.
- `tp1-decode-combined` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round6/tp1-decode-combined-supercollider`.
- Round 6: **39/42 clean assessments now pass** across recorded roots.
  `735d9cb7af7` prevents generated branch-island pools from splitting `s_clause`
  memory runs. This fixes the illegal instructions in D128 block SuperCollider,
  D128 pressure Default, and TP1 decode Default. Both modes of all three
  workloads pass with the rebuilt hook and maintained deadlines. The reduced
  D128 diagnostic reproduced the failure even when the failing kernel received
  no sanitizer probes, exposing the relocation defect. All 382 translator tests
  (including a new RDNA4 clause-preservation regression) and 1,055 host ConSan
  tests pass. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round6/`
  and `d128-debug/` in that campaign root.
  The three remaining clean gaps are both scatter-reduce modes and Default
  top-k (no applicable instrumentation). These counts span hook hashes; final
  qualification must use matching clean/fault configurations. No fault miss is
  promoted by a clean pass; the Qwen 0/8 results remain recorded.
- After the branch-island fix, all **447 physical device tests** pass on
  gfx1201 (serialized, 63.27 seconds). No test failed or was skipped. Log:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/d128-debug/clause-physical.log`.
  This regression gate does not replace external-workload fault qualification.
- Prospective Qwen Default diagnostic: repeat the same reviewed publication
  barrier drop with `RJ_CONSAN_PRESET=max` (every workgroup/cell), eight trials,
  minimum one detection. This tests whether sampling explains the standard
  profile's 0/8 miss. The original miss remains failed qualification; this
  diagnostic requires its own matching clean comparator and cannot promote the
  standard profile. Trial policy is committed before execution.
- Qwen Default dense diagnostic: **7/8 detections**, all numerical oracles
  pass and all trials accepted with pre/post GPU health checks. Sampling every
  workgroup/cell resolves this fault-detection miss in the diagnostic setting.
  The standard profile's 0/8 result remains unchanged; a matching dense clean
  comparator and explicit configuration qualification are still required.
  Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/fault-round2-qwen-dense/`.
- `pytorch-rdna4-llm-topk` / default: missing ConSan analysis verdict; exit -6. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round7/pytorch-rdna4-llm-topk-default`.
- `pytorch-rdna4-llm-topk` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round7/pytorch-rdna4-llm-topk-supercollider`.
- Top-k follow-up: the traced rocPRIM radix-sort kernel shares physical sites
  with other kernel entries. Adding their exact mangled names to a separate
  owner-complete allowlist enables Default instrumentation (45/45 accesses and
  28/28 barriers), but the clean workload now fails with a GPU memory fault.
  SuperCollider still passes with the expanded list. The gap is no longer merely
  inapplicability. Original native lists remain unchanged; closure evidence is
  in `topk-debug/owner-closure.json`, and runs are in `clean-round7/` under
  `/home/benoit/workspace/consan-validation/rdna4-20260923/`.
- Qwen dense clean comparator passes with complete static/dynamic coverage
  (6/6 accesses, 12/12 barriers), required records, no diagnostics, and the
  independent numerical oracle. Its hook, VMFB, inputs, parameters, expected
  outputs, build manifest, allowlist, and tool hashes match the dense fault
  campaign. Effective selection: static/workgroup/cell strides 1, offsets 0,
  legacy-coupled selection; 8 banks per site (48 watchpoints), 128 MiB report
  ceiling, every host epoch, likely group-FLAT provenance, automatic owner
  selection. These results establish a passing clean/fault pair at `preset=max`;
  the standard-profile miss remains yellow. Evidence:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round7/qwen-dense-comparator/`.
- Top-k detailed retry reproduces `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`.
  Pristine and patched code objects are captured in `topk-debug/default-closure/`.
  The subsequent native gfx1201 smoke passes all 1,048,576 values across 4,096
  workgroups; the device remains usable. Evidence:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/topk-debug/post-fault-smoke.log`.
- Validation runner preset integration: all **194 Python tests** pass. Log:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/preset-runner-tests.log`.
- `pytorch-rdna4-llm-topk` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round8/pytorch-rdna4-llm-topk-default`.
- `pytorch-rdna4-llm-topk` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round8/pytorch-rdna4-llm-topk-supercollider`.
- Round 8: **40/42 clean assessments pass** across recorded configurations.
  Top-k Default and SuperCollider both pass with the owner-complete allowlist.
  Fix `60b0a2c5585` emits one compatible entry prologue per physical kernel
  entry while retaining every descriptor owner. Previously, eight aliases
  executed eight consecutive register-remapping prologues, corrupting the
  kernarg pointer and causing the memory fault. Incompatible prologues now
  fail closed. The new alias regression and all 60 related shared/alias tests
  pass. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round8/`
  and `topk-debug/alias-tests.log` in that campaign root. Only the two
  scatter-reduce modes still lack accepted clean evidence.
- New fault inventories are complete for WMMA attention (36 sites / 18 barrier
  sequences) and scatter-reduce (16 atomic sites per weakening family).
  These are static candidates requiring review, not fault-detection passes.
  Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/inventory-round2/`.
- Alias-fix regression gates: all **1,056 host ConSan tests** and **447 physical
  device tests** pass with the rebuilt hook. Physical execution was serialized.
  Logs: `topk-debug/fix-host-tests.log` and `topk-debug/fix-physical-tests.log`
  under `/home/benoit/workspace/consan-validation/rdna4-20260923/`.
  These gates preserve the 40 clean passes as campaign evidence; external
  reviewed-fault qualification still needs to be completed against the final
  hook and each cell's recorded configuration.
- WMMA fault review: `sampled_watchpoint_context::barrier()` executes two
  `__syncthreads()` calls around `++epoch_`. The current FastContext ISA confirms
  paired physical barriers around publication (for example .text+0x13234/0x13258
  followed by 0x1326c/0x13270). Removing just one signal/wait pair retains a
  barrier before the consumer and is not an adequate missing-publication fault.
  The initial barrier belongs to context initialization. No such single-pair
  trial is counted as sensitivity qualification. A reviewed logical-barrier
  mutation or another meaningful fault is needed. Evidence: source
  `hip-moi/include/hip_moi/sampled_watchpoint_context.hpp` and
  `/home/benoit/workspace/consan-validation/rdna4-20260923/wmma-debug/original.asm`.
- Precommitted torch.mode fault: drop the sole signal/wait publication pair
  at .text+0x351e0/0x351e4 between cross-wave stride-64 swaps and the next
  stride-32 compare stage. Reviewed ISA proves the pair is reached after EXEC
  restoration; the row-bounds exit is false for the dispatched input. Require
  at least one detection in eight trials per mode; SuperCollider delays cycle
  through 0/4/16/64 NOPs twice. Independent numerical outcomes remain separate.
  Review artifact: `/home/benoit/workspace/consan-validation/rdna4-20260923/torch-mode-debug/mode.asm`.
- torch.mode Default reviewed fault: **0/8 detections**, below the precommitted
  minimum of one. All eight mutations were admitted, all numerical oracles
  passed, all exits were normal, and every pre/post health check passed. Reach
  is supported by the reviewed unconditional ISA path. Evidence:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/fault-round3-mode-default/`.
  Precommitted a separate eight-trial `preset=max` diagnostic with the same
  site and minimum detection requirement; standard sampling remains unqualified.
- torch.mode SuperCollider exploratory trial set: 0/8 detections and 0/8
  numerical failures. The selected fault policy was unchanged, but the shared
  spec file gained the separate dense policy while this run was active. The
  old runner hashed that file at completion rather than at load time, so this
  root is retained as exploratory and is being rerun with stable provenance.
  The runner now saves the loaded bytes and rejects mismatched
  saved specs on resume; all **195 validation-runner tests** pass. Evidence:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/fault-round3-mode-supercollider/`
  and `fault-snapshot-tests.log` in that campaign root.
- Scatter-reduce review: both dtypes use global-memory atomics (BF16 packed
  add/CAS and FP32 add), with **zero LDS or group-FLAT access sites** in the
  selected code objects. All 16 atomic candidates have conservative/unknown
  ordering roles. Default ordering observation requires a selected directional
  LDS/group-FLAT access window; SuperCollider treats atomic mutations as
  mutation-only and has no repeated-read coverage for this workload's global
  accesses. Thus these rows currently lack applicable detector coverage; the
  collision-count numerical oracle alone cannot qualify them. Keep both cells
  non-green. Inventory evidence:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/inventory-round2/pytorch-scatter-reduce/`.
- torch.mode SuperCollider stable-spec rerun: **0/8 detections**, failing
  the precommitted minimum of one. All eight trials were admitted/reached;
  numerical oracles and pre/post health checks passed. The saved spec snapshot
  fixes the earlier provenance issue, but sensitivity remains unqualified.
  Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/fault-round4-mode-supercollider/`.
- `pytorch-torch-mode` / Default dense diagnostic: **8/8 detections**
  with `RJ_CONSAN_PRESET=max`; all trials admitted and reached the reviewed
  publication fault, all numerical oracles passed, and all before/after GPU
  health and HIP smoke checks passed. The matching dense clean comparator
  passes with 244/244 access and 102/102 barrier sites covered and complete
  dynamic analysis. Both use hook SHA256
  `70aabce67d009663ac06cce576751563dd1be9370974c95081e7a168842a0c46`.
  Standard Default and SuperCollider remain 0/8; this separate configuration
  does not promote either standard cell to green.
  Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/fault-round4-mode-dense/`
  and `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round9/mode-dense-comparator/`.
- Round 10 begins a clean regression sweep of all 42 cells using the current
  hook above, release hip-moi binaries, native trace allowlists, and the
  reviewed top-k owner-expanded allowlist. Each completed cell is recorded
  below; earlier fault assessments remain visible.
- `rdna4-matmul-fp16-production` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/rdna4-matmul-fp16-production-default`.
- `rdna4-matmul-fp16-production` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/rdna4-matmul-fp16-production-supercollider`.
- WMMA follow-up review: the current hook already supports a grouped drop
  through exact primary and companion barrier-sequence identities. Precommitted
  `barrier-drop-kv-publication-group` selects both FastContext K/V publication
  pairs (`.text+0x13234/0x13258` and `0x1326c/0x13270`) as one logical mutation.
  Both profiles require at least one detection in eight trials. Spec loading
  passes; live trials are pending the serialized clean sweep. The earlier
  single-pair review remains valid, but no new mutation mechanism is needed.
- WMMA grouped-drop preparation: both existing host regressions pass
  (`FaultDropBarrierExactGroup*`), including four physical rewrites accounted
  as one mutation and rejection of duplicate/reversed/partial groups without
  writes. The current hook contains both companion controls, and the Python
  runner retains all identity controls and checks one applied logical mutation.
  This is preparation evidence, not physical fault qualification.
  Log: `/home/benoit/workspace/consan-validation/rdna4-20260923/wmma-debug/grouped-drop-tests.log`.
- `rdna4-matmul-fp8-production` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/rdna4-matmul-fp8-production-default`.
- Production FP16 fault review: current-library offline inventory confirms
  ELF `fnv1a64:0ffc359899105b1d`, initial-tile publication signal/wait
  `.text+0xc38/0xc88`. Both validation K sizes take this multi-group path;
  NoopPolicy emits one physical pair for this logical edge. Precommitted
  `barrier-drop-initial-tile-publication` requires at least one detection in
  eight trials per mode. Live trials pending; both current-build clean cells
  pass. Review evidence:
  `/home/benoit/workspace/consan-validation/rdna4-20260923/matmul-debug/`.
- `rdna4-matmul-fp8-production` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/rdna4-matmul-fp8-production-supercollider`.
- `qwen-prefill` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/qwen-prefill-default`.
- `qwen-prefill` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/qwen-prefill-supercollider`.
- `pytorch-torch-mode` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-torch-mode-default`.
- `pytorch-torch-mode` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-torch-mode-supercollider`.
- `tp1-prefill` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/tp1-prefill-default`.
- `tp1-prefill` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/tp1-prefill-supercollider`.
- `tp1-decode-combined` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/tp1-decode-combined-default`.
- `tp1-decode-combined` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/tp1-decode-combined-supercollider`.
- `pytorch-scatter-reduce` / default: no applicable code object; analysis incomplete. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-scatter-reduce-default`.
- `pytorch-scatter-reduce` / supercollider: no applicable code object; analysis incomplete. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-scatter-reduce-supercollider`.
- `pytorch-rdna4-compiled-softmax` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-rdna4-compiled-softmax-default`.
- `pytorch-rdna4-compiled-softmax` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-rdna4-compiled-softmax-supercollider`.
- Production FP8 fault review: current-library offline inventory confirms
  ELF `fnv1a64:0ffc359899105b1d`, initial-tile publication signal/wait
  `.text+0x3f2200/0x3f2260`. Both validation K sizes execute this edge.
  Precommitted eight trials per mode, requiring at least one detection.
  WMMA and then FP16/FP8 fault trials are queued after the complete round-10
  clean sweep; each completed fault cell will be recorded separately.
  Review: `/home/benoit/workspace/consan-validation/rdna4-20260923/matmul-debug/fp8-inventory.txt`.
- `pytorch-rdna4-split-softmax` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-rdna4-split-softmax-default`.
- `pytorch-rdna4-split-softmax` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-rdna4-split-softmax-supercollider`.
- `pytorch-rdna4-llm-topk` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-rdna4-llm-topk-default`.
- `pytorch-rdna4-llm-topk` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-rdna4-llm-topk-supercollider`.
- `llama-rdna4-mul-mat-vec-q` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/llama-rdna4-mul-mat-vec-q-default`.
- `llama-rdna4-mul-mat-vec-q` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/llama-rdna4-mul-mat-vec-q-supercollider`.
- `tp2-family` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/tp2-family-default`.
- `tp2-family` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/tp2-family-supercollider`.
- `clip-bf16` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/clip-bf16-default`.
- `clip-bf16` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/clip-bf16-supercollider`.
- `pytorch-torch-histc` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-torch-histc-default`.
- `pytorch-torch-histc` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/pytorch-torch-histc-supercollider`.
- `llama-rdna4-rms-norm` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/llama-rdna4-rms-norm-default`.
- `llama-rdna4-rms-norm` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/llama-rdna4-rms-norm-supercollider`.
- `d128-block` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/d128-block-default`.
- `d128-block` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/d128-block-supercollider`.
- `d128-pressure` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/d128-pressure-default`.
- `d128-pressure` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/d128-pressure-supercollider`.
- `wmma-attention` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/wmma-attention-default`.
- `wmma-attention` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/wmma-attention-supercollider`.
- `streamk-arrival` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/streamk-arrival-default`.
- `streamk-arrival` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/streamk-arrival-supercollider`.
- `tree-atomic-or` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/tree-atomic-or-default`.
- `tree-atomic-or` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/tree-atomic-or-supercollider`.
- `jakub-attention` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/jakub-attention-default`.
- `jakub-attention` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round10/jakub-attention-supercollider`.
- `wmma-attention` / default: grouped K/V publication fault detects 0/8; admitted 8, reached 8; aggregate accepted=False. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/fault-round5-wmma-default`. Final qualification requires comparator/provenance audit.
- `wmma-attention` / supercollider: grouped K/V publication fault detects 0/8; admitted 8, reached 8; aggregate accepted=False. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/fault-round5-wmma-supercollider`. Final qualification requires comparator/provenance audit.
