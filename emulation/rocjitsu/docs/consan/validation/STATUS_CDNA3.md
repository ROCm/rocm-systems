# ConSan CDNA3 (`gfx942`) status

Historical evidence: this ledger was not requalified during the September 2026
documentation audit. Existing results describe their recorded runs, not the
current branch. Rerun the linked procedure after relevant changes.

Start any new revalidation with
[rocprofv3-based allowlist discovery and application](VALIDATION.md#first-step-for-revalidation-generate-and-apply-kernel-allowlists).
Apply this before retrying recorded timeouts or raising deadlines; update their
status only after new runs provide evidence.

Physical `gfx942` qualification is unavailable in the active workspace. The
hip-moi rows have target-native simulator prerequisite coverage through
RocJITsu `configs/gfx942_cdna3_kmd.json`; simulator evidence does not promote a
physical E2E cell. Refresh this ledger after relevant source, toolchain,
workload, emulator, or runtime changes.

Shared [color scale](VALIDATION.md#status-colors): 🟩 qualified; 🟨 clean run
established, but fault qualification is pending/below bar or the workload is
outside detector scope; 🟧 clean qualification blocked by prerequisites,
unsupported applicable operations, incomplete coverage/evidence, or a timeout;
🟥 observed correctness or instrumentation failure. Empty/🩶 means unassessed
for this execution target; simulator prerequisites alone do not qualify hardware.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🩶 unassessed | 🩶 unassessed |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🩶 unassessed | 🩶 unassessed |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🩶 unassessed | 🩶 unassessed |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🩶 unassessed | 🩶 unassessed |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🩶 unassessed | 🩶 unassessed |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| Main E2E | P4 | hip-moi MFMA attention (`wmma-attention`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
