# ConSan RDNA4 (`gfx1201`) status

Historical evidence: this ledger was not requalified during the September 2026
documentation audit. Existing results describe their recorded runs, not the
current branch. Rerun the linked procedure after relevant changes.

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
The table below retains historical results until each cell has fresh evidence;
none of those historical colors count as a pass for this campaign.

- Host: AMD Radeon RX 9070 (`gfx1201`); native HIP shared-memory smoke passed.
- SDK: `/home/benoit/venv`, ROCm `10.2.0a20260915` development package.
- Preparation: current hook and production matmul built; generated per-workload
  allowlists applied; Qwen build provenance verified through the campaign symlinks.
- Fresh clean assessments: **42/42**; no cell has passed fresh fault qualification.
- Preparation logs: `/home/benoit/workspace/consan-validation-artifacts/`.

Legend: 🩶 unseen · 🟥 broken before useful evidence · 🟧 below 80% aggregate applicable-site support or another substantial gap · 🟨 timeout-only blocker or at least 80% aggregate applicable-site support · 🟩 accepted workload/profile contract.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Production HIP | P0 | FP16 matmul (`rdna4-matmul-fp16-production`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending (timeout override 300s) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending (timeout override 300s) |
| Production HIP | P0 | FP8 matmul (`rdna4-matmul-fp8-production`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending (timeout override 300s) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending (timeout override 300s) |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟨 Sep 23: clean pass; reviewed publication-barrier fault missed in 8/8 trials | 🟨 Sep 23: clean pass; reviewed publication-barrier fault missed in 8/8 trials |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟥 Sep 23: illegal instruction, then teardown timeout at 300s; native oracle passes | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending (timeout override 300s) |
| PyTorch | P1 | collision-heavy `scatter_reduce` (`pytorch-scatter-reduce`) | 🟥 Sep 23: no applicable code object; analysis incomplete | 🟥 Sep 23: no applicable code object; analysis incomplete |
| PyTorch | P2 | Inductor compiled softmax (`pytorch-rdna4-compiled-softmax`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P2 | split online softmax (`pytorch-rdna4-split-softmax`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P2 | Qwen-vocabulary top-k (`pytorch-rdna4-llm-topk`) | 🟥 Sep 23: no applicable code object; analysis incomplete | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| llama.cpp | P2 | quantized matvec (`llama-rdna4-mul-mat-vec-q`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P3 | native histogram (`pytorch-torch-histc`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| llama.cpp | P3 | RMS norm (`llama-rdna4-rms-norm`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟥 Sep 23: Release native pass; Default illegal instruction in WideKey SampledFast test | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
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
- Latest clean assessments: 36/42 pass across the recorded artifact roots.
  Six gaps remain: both scatter-reduce modes and Default top-k are inapplicable;
  Default TP1 decode hits an illegal instruction before teardown times out; optimized D128 block SuperCollider and D128
  pressure Default hit illegal instructions in sampled-fast cases. The latter
  are ordinary numeric-reference tests, not intentional fault cases, and remain
  in the clean workload filters. Other optimized hip-moi cells pass. This
  count includes earlier hook hashes; final qualification must match the tested
  binary/configuration and complete the reviewed fault checks.
- `tp1-decode-combined` / default: missing ConSan analysis verdict; exit 124. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round5/tp1-decode-combined-default`.
- `tp1-decode-combined` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round5/tp1-decode-combined-supercollider`.
- TP1 decode Default retry with the repaired hook reproduces `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION`, followed by a 300-second teardown timeout. Its native baseline and SuperCollider run pass. A longer deadline does not resolve this failure. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round5/`.
- `d128-block` / default: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round6/d128-block-default`.
- `d128-block` / supercollider: fresh clean pass; fault qualification pending. Evidence: `/home/benoit/workspace/consan-validation/rdna4-20260923/clean-round6/d128-block-supercollider`.
