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
- Fresh clean assessments: **11/42**; no cell has completed fresh fault qualification.
- Preparation logs: `/home/benoit/workspace/consan-validation-artifacts/`.

Legend: 🩶 unseen · 🟥 broken before useful evidence · 🟧 below 80% aggregate applicable-site support or another substantial gap · 🟨 timeout-only blocker or at least 80% aggregate applicable-site support · 🟩 accepted workload/profile contract.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Production HIP | P0 | FP16 matmul (`rdna4-matmul-fp16-production`) | 🩶 unassessed | 🩶 unassessed |
| Production HIP | P0 | FP8 matmul (`rdna4-matmul-fp8-production`) | 🟩 exact; 227/227 accesses and 48/48 barriers; reviewed fault bundle | 🟧 exact; only 75/227 accesses supported |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟩 exact; 20/20 accesses and 26/26 barriers; 17/32 fault sweep | 🟩 exact; 20/20 accesses; fault bundle |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟩 exact; 352/352 accesses and 86/86 barriers; fault bundle | 🟩 exact; 352/352 accesses; fault bundle |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟩 exact; 704/704 accesses and 172/172 barriers; fault bundle | 🟩 exact; 704/704 accesses; fault bundle |
| PyTorch | P1 | collision-heavy `scatter_reduce` (`pytorch-scatter-reduce`) | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log |
| PyTorch | P2 | Inductor compiled softmax (`pytorch-rdna4-compiled-softmax`) | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log |
| PyTorch | P2 | split online softmax (`pytorch-rdna4-split-softmax`) | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log | 🟩 exact; 8/8 accesses; fault bundle |
| PyTorch | P2 | Qwen-vocabulary top-k (`pytorch-rdna4-llm-topk`) | 🟨 exact/complete; 418,292 accesses and 100,916 barriers; reviewed-fault refresh pending | 🟨 exact/complete; 58,992/58,992 accesses; reviewed-fault refresh pending |
| llama.cpp | P2 | quantized matvec (`llama-rdna4-mul-mat-vec-q`) | 🟩 exact; 462 accesses and 88 barriers; fault bundle | 🟩 exact; 462/462 accesses; fault bundle |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🟩 exact; 2,976 accesses and 420 barriers; fault bundle | 🟩 exact; 2,976/2,976 accesses; fault bundle |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟩 exact; 85 accesses and 72 barriers; fault bundle | 🟩 exact; 85/85 accesses; fault bundle |
| PyTorch | P3 | native histogram (`pytorch-torch-histc`) | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log | 🟥 Sep 23: native rocprofv3 discovery failed; inspect discovery log |
| llama.cpp | P3 | RMS norm (`llama-rdna4-rms-norm`) | 🟨 exact/complete; effective reviewed fault pending | 🟨 exact/complete; effective reviewed fault pending |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟥 Sep 23: native oracle passes; strict load rejects relocation of `s_swappc_b64` at `.text+121248`; fix in progress | 🟥 Sep 23: strict load rejects the same indirect-call relocation; fix in progress |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 exact; 12 accesses and 8 barriers; fault bundle | 🟩 exact; 12/12 accesses; fault bundle |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 exact; 12 accesses and 8 barriers; fault bundle | 🟩 exact; 12/12 accesses; fault bundle |
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
