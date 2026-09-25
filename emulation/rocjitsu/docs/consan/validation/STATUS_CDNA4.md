# ConSan validation on CDNA4 (gfx950)

Colors match [STATUS_RDNA4.md](STATUS_RDNA4.md): 🟩 qualified (matching clean
pass, complete coverage, healthy GPU, fault detections ≥6/8); 🟨 pending, below
bar, or out of scope; 🟥 correctness or instrumentation failure. Fault counts
are detections. “Lowest passing” requires all lower presets to have failed.
[Procedure](VALIDATION.md) · [Evidence and repairs](CDNA4_DEFAULT_REPAIRS_20260924.md)

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Main E2E | P2 | Sharktank TP2 prefill/decode/combined (`tp2-family`, `tp2-decode`, `tp2-combined`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8); supporting decode/combined clean pass | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar; supporting decode/combined clean pass |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟩 high (lowest passing): clean pass; fault 7/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟩 sleep=15: clean pass; fault 6/8 (bar 6/8) |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Main E2E | P4 | hip-moi MFMA attention (`wmma-attention`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟩 sleep=15: clean pass; fault 8/8 (bar 6/8) |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 higher (lowest passing): repaired publication journal and release fault; clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟨 repaired hip-moi atomic cache; max clean 3/3; default/high fault 0/8, 1/8; higher running | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P0 | HIP matmul 128 cubed (`hip-matmul-m128-n128-k128`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P0 | HipKittens BF16 (`hipkittens-bf16fp32-16x32`) | 🟨 max: clean pass; fault 0/8; fresh 64/128/256/512-bank trials each 0/8; below bar | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P1 | HipKittens FP8 (`hipkittens-fp8fp32-4wave`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P1 | HipKittens MXFP8 (`hipkittens-mxfp8-4wave`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P1 | HIP Stream-K simple (`hip-streamk-simple-m256-n256-k256`) | 🟩 high (lowest passing): clean pass; fault 7/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P1 | HIP Stream-K two-tile (`hip-streamk-two-tile-m256-n256-k256`) | 🟩 high (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P2 | rocBLAS SGEMM square-64 (`rocblas-sgemm-square-64`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Tensile | P0 | gfx950 LDS-positive BF16 GEMM (`tensile-gfx950-lds-positive`) | 🟨 max: clean pass; fault 0/8; fresh 64/128/256/512-bank trials each 0/8; below bar | 🟩 delay-zero: clean pass; fault 8/8 (bar 6/8) |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 1/8 (bar 6/8); below bar |
| PyTorch | P0 | `torch.topk` (`pytorch-torch-topk`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| PyTorch | P1 | `torch.sort` (`pytorch-torch-sort`) | 🟩 higher (lowest passing): clean pass; fault 7/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟨 max: clean pass; fault 0/8; fresh 64/128/256/512-bank trials each 0/8; below bar | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| PyTorch | P2 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟨 Out of scope: numerical pass; traced global-atomic kernels have no applicable LDS/FLAT race coverage | 🟨 Out of scope: numerical pass; traced global-atomic kernels have no applicable LDS/FLAT race coverage |
| PyTorch | P2 | norm/softmax (`pytorch-norm-softmax`) | 🟨 max: clean pass; fault 0/8; 64/128/256/512-bank trials each 0/8; below bar | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
