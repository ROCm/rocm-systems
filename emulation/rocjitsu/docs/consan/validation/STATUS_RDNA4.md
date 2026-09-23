| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Production HIP | P0 | FP16 matmul (`rdna4-matmul-fp16-production`) | 🟩 higher: clean pass; fault 8/8 (bar 6/8) | 🟨 Sep 23: clean pass; fault 0/8 detections; 8 trials admitted after transform fix |
| Production HIP | P0 | FP8 matmul (`rdna4-matmul-fp8-production`) | 🟩 higher: clean pass; fault 8/8 (bar 6/8) | 🟩 Sep 23: clean pass; fault 1/8 (delay=0) |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟨 higher: clean pass; fault trials running | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending; prior fault assessment: clean pass; reviewed publication-barrier fault missed in 8/8 trials |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending; prior fault assessment: standard fault 0/8; preset=max clean passes and fault detects 8/8 (separate configuration) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending; prior fault assessment: clean pass; reviewed cross-wave publication fault missed in 8/8 trials |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟨 Clean pass; reviewed fault ready; lowest-preset search queued | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟨 Clean pass; reviewed fault ready; lowest-preset search queued | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P1 | collision-heavy `scatter_reduce` (`pytorch-scatter-reduce`) | 🟥 Sep 23: no applicable code object; analysis incomplete | 🟥 Sep 23: no applicable code object; analysis incomplete |
| PyTorch | P2 | Inductor compiled softmax (`pytorch-rdna4-compiled-softmax`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P2 | split online softmax (`pytorch-rdna4-split-softmax`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P2 | Qwen-vocabulary top-k (`pytorch-rdna4-llm-topk`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| llama.cpp | P2 | quantized matvec (`llama-rdna4-mul-mat-vec-q`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| PyTorch | P3 | native histogram (`pytorch-torch-histc`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| llama.cpp | P3 | RMS norm (`llama-rdna4-rms-norm`) | 🟩 higher: clean pass; fault 8/8 (bar 6/8) | 🟨 Sep 23: clean pass; publication fault 0/8 detections; fault qualification failed |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟨 Sep 23: clean pass; publication fault 0/8 detections; fault qualification failed | 🟨 Sep 23: clean pass; publication fault 0/8 detections; fault qualification failed |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 higher: clean pass; fault 8/8 (bar 6/8) | 🟨 Sep 23: clean pass; grouped publication fault 0/8 detections; fault qualification failed |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending | 🟨 Sep 23: fresh clean pass; reviewed fault qualification pending |
