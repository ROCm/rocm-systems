# ConSan CDNA3 (`gfx942`) status

Status snapshot: 2026-08-24. Physical gfx942 qualification is unavailable in this workspace. The six hip-moi rows pass their target-native simulator prerequisites through RocJitsu `configs/gfx942_cdna3_kmd.json`, but simulator evidence does not promote a physical E2E cell.

Legend: 🩶 unseen or simulator-prerequisite-only · 🟥 broken before useful evidence · 🟧 below 80% aggregate applicable-site support or another substantial gap · 🟨 timeout-only blocker or at least 80% aggregate applicable-site support · 🟩 accepted workload/profile contract.

| Set | Priority | Workload / validation ID | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---:|---|---|---|---|---|
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| Main E2E | P4 | hip-moi MFMA attention (`wmma-attention`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
