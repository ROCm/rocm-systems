# ConSan RDNA3 (`gfx1100`) status

Historical evidence: this ledger was not requalified during the September 2026
documentation audit. Existing results describe their recorded runs, not the
current branch. Rerun the linked procedure after relevant changes.

Start any new revalidation with
[rocprofv3-based allowlist discovery and application](VALIDATION.md#first-step-for-revalidation-generate-and-apply-kernel-allowlists).
Apply this before retrying recorded timeouts or raising deadlines; update their
status only after new runs provide evidence.

Physical evidence uses native `gfx1100` code objects on a matching GPU;
simulator prerequisites use RocJITsu `configs/gfx1100_w7900.json`. Exact counts
belong to the binaries used for qualification and must be refreshed after
relevant source, toolchain, workload, emulator, or runtime changes.

Shared [color scale](VALIDATION.md#status-colors): 🟩 qualified; 🟨 clean run
established, but fault qualification is pending/below bar or the workload is
outside detector scope; 🟧 clean qualification blocked by prerequisites,
unsupported applicable operations, incomplete coverage/evidence, or a timeout;
🟥 observed correctness or instrumentation failure. Empty/🩶 means unassessed
for this execution target; simulator prerequisites alone do not qualify hardware.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Physical compact gate | P0 | native two-wave LDS fixture (`ConSanGfx1100Physical.*`) | 🟨 exact clean output, visible records, zero diagnostics; reviewed conflict and broad campaign missing | 🟨 exact clean/all-sites rows and mutation containment; broad E2E fault campaign missing |
| Simulator prerequisite | P0 | native two-wave LDS fixture (`ConSanGfx1100Sim.*`) | 🩶 prerequisite passes | 🩶 prerequisite passes |
| Broad E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi D128 block (`d128-block`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi MFMA attention (`wmma-attention`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🩶 unassessed | 🩶 unassessed |
