# ConSan RDNA3 (`gfx1100`) status

Status snapshot: 2026-08-24. Physical evidence uses native gfx1100 code objects on the Radeon Pro W7900 selected by agent UUID; simulator prerequisites use RocJitsu `configs/gfx1100_w7900.json`.

Legend: 🩶 unseen or simulator-prerequisite-only · 🟥 broken before useful evidence · 🟧 below 80% aggregate applicable-site support or another substantial gap · 🟨 timeout-only blocker or at least 80% aggregate applicable-site support · 🟩 accepted workload/profile contract.

| Set | Priority | Workload / validation ID | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---:|---|---|---|---|---|
| Physical compact gate | P0 | native two-wave LDS fixture (`ConSanGfx1100Physical.*`) | 🟨 exact clean/all-sites rows and mutation containment; broad E2E fault/overhead campaign missing | 🟨 exact clean output, visible records, zero diagnostics; reviewed conflict and broad campaign missing | 🟨 exact clean output, visible records, zero diagnostics; reviewed conflict and broad campaign missing | 🟨 exact clean output plus attributed two-wave conflict; broad E2E overhead/fault campaign missing |
| Simulator prerequisite | P0 | native two-wave LDS fixture (`ConSanGfx1100Sim.*`) | 🩶 prerequisite passes | 🩶 prerequisite passes | 🩶 prerequisite passes | 🩶 clean and conflict prerequisites pass |
| Broad E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi D128 block (`d128-block`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi MFMA attention (`wmma-attention`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
