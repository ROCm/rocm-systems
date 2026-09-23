# ConSan CDNA4 (`gfx950`) status

Historical evidence: this ledger was not requalified during the September 2026
documentation audit. Existing results describe their recorded runs, not the
current branch. Rerun the linked procedure after relevant changes.

Start any new revalidation with
[rocprofv3-based allowlist discovery and application](VALIDATION.md#first-step-for-revalidation-generate-and-apply-kernel-allowlists).
Apply this before retrying recorded timeouts or raising deadlines; update their
status only after new runs provide evidence.

This ledger summarizes the latest accepted physical or explicitly labeled
simulator evidence. Exact counts belong to the native binaries used for those
runs and must be refreshed after relevant source, toolchain, workload, or
runtime changes. Procedures and acceptance rules are in
[VALIDATION.md](VALIDATION.md).

Legend: 🩶 unseen · 🟥 broken before useful evidence · 🟧 below 80% aggregate applicable-site support or another substantial gap · 🟨 timeout-only blocker or at least 80% aggregate applicable-site support · 🟩 accepted workload/profile contract.

| Set | Priority | Workload / validation ID | SuperCollider | ConSan |
| --- | ---: | --- | --- | --- |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟩 exact; 844/844 accesses | 🟩 exact; 844/844 accesses and 37/37 barriers |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟩 physical; exact/complete; 176/176 accesses | 🟩 physical; exact; 176/176 accesses and 28/28 barriers |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟩 physical; exact/complete; 352/352 accesses | 🟩 physical; exact; 352/352 accesses and 56/56 barriers |
| Main E2E | P2 | Sharktank TP2 prefill/decode/combined (`tp2-family`, `tp2-decode`, `tp2-combined`) | 🟩 physical; three exact oracles; each row has 544/544 accesses | 🟩 physical; three exact oracles; each row has 544/544 accesses and 60/60 barriers |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟩 physical; exact; 45/45 accesses | 🟩 physical; exact; 45/45 accesses and 24/24 barriers |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 physical; exact; 122/122 accesses | 🟩 physical; exact; 122/122 accesses and 119/119 barriers |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 physical; four exact oracles; 236/236 accesses; paired/fault bundle | 🟩 physical; exact; 236/236 accesses and 28/28 barriers; paired/fault bundle |
| Main E2E | P4 | hip-moi MFMA attention (`wmma-attention`) | 🟩 physical; exact; 50/50 accesses; paired/fault bundle | 🟩 physical; exact; 50/50 accesses and 14/14 barriers; paired/fault bundle |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 physical; exact; 32/32 accesses | 🟩 physical; exact/complete; 32/32 accesses, 6/6 barriers, 1/1 atomic, 2/2 fences |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 physical; exact; 48/48 accesses | 🟩 physical; exact/complete; 48/48 accesses, 6/6 barriers, 3/3 atomics, 5/5 fences |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟩 four exact oracles; 338/338 accesses | 🟩 exact; 338/338 accesses and 35/35 barriers |
| Test corpus | P0 | HIP matmul 128 cubed (`hip-matmul-m128-n128-k128`) | 🟩 physical; exact; 739/739 accesses; qualified exact-one FP16 tile-publication miss | 🟩 physical; exact; 739/739 accesses and 109/109 barriers |
| Test corpus | P0 | HipKittens BF16 (`hipkittens-bf16fp32-16x32`) | 🟩 physical; exact; 128/128 accesses; qualified exact-one prologue-publication miss | 🟩 physical; exact/complete; 128/128 accesses and 32/32 barriers |
| Test corpus | P1 | HipKittens FP8 (`hipkittens-fp8fp32-4wave`) | 🟩 exact; 64/64 accesses; qualified exact-one prologue-publication miss | 🟩 exact; 96/96 accesses and 5/5 barriers |
| Test corpus | P1 | HipKittens MXFP8 (`hipkittens-mxfp8-4wave`) | 🟩 physical; exact; 96/96 accesses; qualified exact-one prologue-publication miss | 🟩 physical; exact/complete; 96/96 accesses and 5/5 barriers |
| Test corpus | P1 | HIP Stream-K simple (`hip-streamk-simple-m256-n256-k256`) | 🟩 physical; exact; 32/32 accesses; qualified exact-one initial-tile publication miss | 🟩 physical; exact/complete; 32/32 accesses, 3/3 barriers, 2/2 atomics, and 2/2 fences |
| Test corpus | P1 | HIP Stream-K two-tile (`hip-streamk-two-tile-m256-n256-k256`) | 🟩 physical; exact; 80/80 accesses; qualified exact-one initial-tile publication miss | 🟩 physical; exact/complete with zero diagnostics; 80/80 accesses, 5/5 barriers, 2/2 atomics, and 2/2 fences |
| Test corpus | P2 | rocBLAS SGEMM square-64 (`rocblas-sgemm-square-64`) | 🟩 physical exact/complete; 49,435/49,435 accesses; qualified exact-one initial-tile publication miss | 🟩 physical exact/complete; 49,435/49,435 accesses and 4,997/4,997 barriers; qualified exact-one publication miss |
| Tensile | P0 | gfx950 Stream-K SGEMM (`tensile-gfx950-lds-positive`) | 🟩 physical exact/complete; 48/48 accesses and 9/9 barriers; exact-binary wrong-address fault diagnosed | 🟩 physical exact/complete; 48/48 accesses and 9/9 barriers; qualified exact-binary wrong-address miss |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 physical exact/complete after exact relocation-owned descriptor proof; 25,366/25,366 accesses | 🟨 physical exact oracle and dynamic analysis complete; static coverage 25,704/26,426 accesses and 2,659/4,359 barriers |
| PyTorch | P0 | `torch.topk` (`pytorch-torch-topk`) | 🟩 gfx950 emulation exact/complete; 239,442/239,442 accesses | 🟩 gfx950 emulation exact/complete with zero diagnostics; 239,730/239,730 accesses and 11,423/11,423 barriers |
| PyTorch | P1 | `torch.sort` (`pytorch-torch-sort`) | 🟩 physical exact/complete; 53,064/53,064 accesses | 🟨 physical run reaches the 45-s cap during main-object transformation; emulator run is exact/complete with qualified-fault evidence |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟩 physical FP32/FP64 exact/complete; 110/110 accesses | 🟩 physical FP32/FP64 exact/complete; 152/152 accesses and 84/84 barriers |
| PyTorch | P2 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟩 physical BF16/FP32 exact/complete; 27/27 accesses | 🟩 physical BF16/FP32 exact/complete; 27/27 accesses |
| PyTorch | P2 | norm/softmax (`pytorch-norm-softmax`) | 🟩 physical exact/complete; 4,880/4,880 accesses | 🟩 physical exact/complete; 4,880/4,880 accesses and 2,132/2,132 barriers |
