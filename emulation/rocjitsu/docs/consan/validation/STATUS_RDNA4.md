| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Production HIP | P0 | FP16 matmul (`rdna4-matmul-fp16-production`) | 🟩 higher (lowest verified): clean pass; fault 8/8 (bar 6/8) | 🟨 delay-matrix: clean pass; fault 1/8 (bar 6/8); sleep calibration queued |
| Production HIP | P0 | FP8 matmul (`rdna4-matmul-fp8-production`) | 🟩 higher (lowest verified): clean pass; fault 8/8 (bar 6/8) | 🟨 delay-matrix: clean pass; fault 0/8 (bar 6/8); sleep calibration queued |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 NOP delays: fault 0/8; matching sleep-delay calibration queued |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 NOP delays: fault 0/8; matching sleep-delay calibration queued |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟩 higher (lowest verified): clean pass; fault 8/8 (bar 6/8) | 🟨 Clean pass; reviewed fault ready; delay-matrix qualification queued |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟨 higher: clean pass; fault running | 🟨 Clean pass; reviewed fault ready; delay-matrix qualification queued |
| PyTorch | P1 | collision-heavy `scatter_reduce` (`pytorch-scatter-reduce`) | 🟥 Global-atomic workload; traced kernels have no LDS/FLAT accesses; outside detector scope | 🟥 Global-atomic workload; traced kernels have no LDS/FLAT accesses; outside detector scope |
| PyTorch | P2 | Inductor compiled softmax (`pytorch-rdna4-compiled-softmax`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8); same-value writes allowed | 🟨 delay-matrix: clean pass; fault 1/8 (bar 6/8); sleep calibration queued |
| PyTorch | P2 | split online softmax (`pytorch-rdna4-split-softmax`) | 🟩 high (lowest passing): clean pass; fault 8/8 (bar 6/8); same-value writes allowed | 🟩 delay matrix: fault 8/8; matching clean controls pass |
| PyTorch | P2 | Qwen-vocabulary top-k (`pytorch-rdna4-llm-topk`) | 🟨 Clean pass; reviewed fault ready; lowest-preset search queued | 🟨 Clean pass; reviewed fault ready; delay-matrix qualification queued |
| llama.cpp | P2 | quantized matvec (`llama-rdna4-mul-mat-vec-q`) | 🟩 higher (lowest verified): clean pass; fault 7/8 (bar 6/8) | 🟨 delay-matrix: clean pass; fault 4/8 (bar 6/8); sleep calibration queued |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🟨 Clean pass; reviewed fault ready; lowest-preset search queued | 🟨 Clean pass; reviewed fault ready; delay=0 qualification queued |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟨 Clean pass; reviewed fault ready; lowest-preset search queued | 🟨 Clean pass; reviewed fault ready; delay=0 qualification queued |
| PyTorch | P3 | native histogram (`pytorch-torch-histc`) | 🟨 Clean pass; reviewed fault ready; lowest-preset search queued | 🟨 Clean pass; reviewed fault ready; delay=0 qualification queued |
| llama.cpp | P3 | RMS norm (`llama-rdna4-rms-norm`) | 🟩 higher (lowest verified): clean pass; fault 8/8 (bar 6/8) | 🟨 delay-matrix: clean pass; fault 0/8 (bar 6/8); sleep calibration queued |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 higher (lowest verified): clean pass; fault 8/8 (bar 6/8) | 🟨 delay-matrix: clean pass; fault 0/8 (bar 6/8); sleep calibration queued |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 higher (lowest verified): clean pass; fault 8/8 (bar 6/8) | 🟨 delay-matrix: clean pass; fault 0/8 (bar 6/8); sleep calibration queued |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 high (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 NOP delays: fault 0/8; matching sleep-delay calibration queued |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟨 Clean pass; fault review: redundant LDS waits preserve publication | 🟨 Clean pass; fault review: redundant LDS waits preserve publication |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟨 Clean pass; fault review: redundant LDS waits preserve publication | 🟨 Clean pass; fault review: redundant LDS waits preserve publication |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟨 Clean pass; reviewed fault ready; lowest-preset search queued | 🟨 Clean pass; reviewed fault ready; delay-matrix qualification queued |
