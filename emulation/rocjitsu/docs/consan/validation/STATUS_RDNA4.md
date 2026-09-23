# ConSan validation on RDNA4 (gfx1201)

This table tracks end-to-end ConSan revalidation of the external workloads on
this host’s gfx1201 GPU. **Default** names the sampling preset used by the
Default engine; **SuperCollider** records its replay and timing-perturbation
configuration. Fault counts such as **8/8** mean eight detections in eight
fault-injection trials, not eight passing workloads. **Bar 6/8** means at least
six of eight admitted and reached trials must detect the injected fault. Green
also requires a passing clean correctness run with matching controls, complete
coverage evidence, and healthy GPU checks. “Lowest passing” means all smaller
presets from `default` failed; “lowest verified” means smaller presets remain
untested or unqualified. Yellow is pending or below the qualification bar; red
records a failure or an unsupported workload. See [VALIDATION.md](VALIDATION.md)
for the procedure and qualification rules.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Production HIP | P0 | FP16 matmul (`rdna4-matmul-fp16-production`) | 🟩 high (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
| Production HIP | P0 | FP8 matmul (`rdna4-matmul-fp8-production`) | 🟩 high (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 3/8 (bar 6/8) |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 2/8 (bar 6/8) |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
| PyTorch | P1 | collision-heavy `scatter_reduce` (`pytorch-scatter-reduce`) | 🟥 Global-atomic workload; traced kernels have no LDS/FLAT accesses; outside detector scope | 🟥 Global-atomic workload; traced kernels have no LDS/FLAT accesses; outside detector scope |
| PyTorch | P2 | Inductor compiled softmax (`pytorch-rdna4-compiled-softmax`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8); same-value writes allowed | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); same-value writes allowed |
| PyTorch | P2 | split online softmax (`pytorch-rdna4-split-softmax`) | 🟩 high (lowest passing): clean pass; fault 8/8 (bar 6/8); same-value writes allowed | 🟩 delay matrix: fault 8/8; matching clean controls pass |
| PyTorch | P2 | Qwen-vocabulary top-k (`pytorch-rdna4-llm-topk`) | 🟩 high (lowest passing): clean pass; fault 6/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
| llama.cpp | P2 | quantized matvec (`llama-rdna4-mul-mat-vec-q`) | 🟩 higher (lowest verified): clean pass; fault 7/8 (bar 6/8) | 🟩 sleep=1: clean pass; fault 8/8 (bar 6/8) |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🟩 high (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟩 high (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟩 sleep=1: clean pass; fault 8/8 (bar 6/8) |
| PyTorch | P3 | native histogram (`pytorch-torch-histc`) | 🟨 max + 256 banks: clean pass; fault 0/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
| llama.cpp | P3 | RMS norm (`llama-rdna4-rms-norm`) | 🟩 higher (lowest verified): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 higher (lowest verified): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 higher (lowest verified): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 high (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟥 max: clean numerical pass; ConSan reports 8 cross-wave conflicts | 🟨 delay-zero: clean pass; fault 0/8 (bar 6/8) |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟥 max: clean numerical pass; ConSan reports 12 cross-wave conflicts | 🟨 delay-zero: clean pass; fault 0/8 (bar 6/8) |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8) |
