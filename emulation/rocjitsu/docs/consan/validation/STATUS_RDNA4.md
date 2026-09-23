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
- Preparation: rebuilding the current hook, integrating per-workload generated
  allowlists, preparing Qwen provenance, and building the production matmul.
- Fresh clean assessments: **35/42**; no cell has completed fresh fault qualification.
- Preparation logs: `/home/benoit/workspace/consan-validation-artifacts/`.

Legend: 🩶 unseen · 🟥 broken before useful evidence · 🟧 below 80% aggregate applicable-site support or another substantial gap · 🟨 timeout-only blocker or at least 80% aggregate applicable-site support · 🟩 accepted workload/profile contract.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Production HIP | P0 | FP16 matmul (`rdna4-matmul-fp16-production`) | 🟥 Sep 23: missing ConSan coverage record; exit 124 | 🟥 Sep 23: missing ConSan coverage record; exit 124 |
| Production HIP | P0 | FP8 matmul (`rdna4-matmul-fp8-production`) | 🟥 Sep 23: missing ConSan analysis verdict; exit 124 | 🟥 Sep 23: missing ConSan analysis verdict; exit 124 |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟥 Sep 23: exit 1 | 🟥 Sep 23: exit 1 |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log |
| PyTorch | P1 | collision-heavy `scatter_reduce` (`pytorch-scatter-reduce`) | 🟥 Sep 23: no applicable code object; analysis incomplete | 🟥 Sep 23: no applicable code object; analysis incomplete |
| PyTorch | P2 | Inductor compiled softmax (`pytorch-rdna4-compiled-softmax`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P2 | split online softmax (`pytorch-rdna4-split-softmax`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P2 | Qwen-vocabulary top-k (`pytorch-rdna4-llm-topk`) | 🟥 Sep 23: no applicable code object; analysis incomplete | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| llama.cpp | P2 | quantized matvec (`llama-rdna4-mul-mat-vec-q`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log |
| PyTorch | P3 | native histogram (`pytorch-torch-histc`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| llama.cpp | P3 | RMS norm (`llama-rdna4-rms-norm`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟥 Sep 23: native oracle passes; strict load rejects relocation of `s_swappc_b64` at `.text+121248`; fix in progress | 🟥 Sep 23: strict load rejects the same indirect-call relocation; fix in progress |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟥 Sep 23: ConSan rejected a code object before execution; exit 92 | 🟥 Sep 23: ConSan rejected a code object before execution; exit 92 |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟥 Sep 23: ConSan rejected a code object before execution; exit 92 | 🟩 exact; 12/12 accesses; fault bundle |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 exact; 4 accesses, 15 atomics, 8 barriers; fault bundle | 🟩 exact; 4/4 accesses; fault bundle |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 exact; 4 accesses, 15 atomics, 8 barriers; fault bundle | 🟩 exact; 4/4 accesses; fault bundle |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟩 exact; 31 accesses and 8 barriers; fault bundle | 🟩 exact; 31/31 accesses; fault bundle |

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
