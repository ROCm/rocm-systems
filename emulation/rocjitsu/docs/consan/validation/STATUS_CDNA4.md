# ConSan validation on CDNA4 (gfx950)

This ledger follows [STATUS_RDNA4.md](STATUS_RDNA4.md): **green requires a
matching clean correctness run, complete applicable coverage, healthy GPU
checks, and at least 6 detections in 8 admitted/reached fault trials**.
Fault counts are detections, not passing workloads. Default presets are tested
in ascending order `default`, `high`, `higher`, `max`; “lowest passing” requires
all lower presets to have failed. SuperCollider settings require their own
matching clean controls. Yellow means pending, below the bar, or out of scope;
red means observed correctness or instrumentation failure. See the
[revalidation report](CDNA4_REVALIDATION_20260924.md) for fixes and provenance.

**Physical revalidation, September 24, 2026.** The initial physical
MI350X campaign ran 26 of 27 native workloads and both engines, with 48 accepted
clean profiles across 24 workloads. Clean acceptance alone does not qualify a
green cell. The completed searches extend those runs to the RDNA4 fault bar;
no historical or simulator evidence substitutes for physical qualification.

Every available workload has a native rocprofv3 exact-name allowlist, generated
and applied as specified in [USAGE.md](../USAGE.md#generate-and-use-a-kernel-allowlist)
and [VALIDATION.md](VALIDATION.md). Both modes share each workload's allowlist.
All physical runs use the shared GPU lock and the venv TheRock ROCm stack;
no `/opt/rocm*` installation is used.

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
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟥 higher/max: clean false positives; default/high fault 0/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟥 higher/max: clean false positives; default/high fault 0/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Main E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟨 Unavailable: CDNA4 fixture absent from public hip-moi checkout | 🟨 Unavailable: CDNA4 fixture absent from public hip-moi checkout |
| Test corpus | P0 | HIP matmul 128 cubed (`hip-matmul-m128-n128-k128`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P0 | HipKittens BF16 (`hipkittens-bf16fp32-16x32`) | 🟨 max: clean pass; fault 0/8 (bar 6/8); below bar | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P1 | HipKittens FP8 (`hipkittens-fp8fp32-4wave`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P1 | HipKittens MXFP8 (`hipkittens-mxfp8-4wave`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P1 | HIP Stream-K simple (`hip-streamk-simple-m256-n256-k256`) | 🟩 high (lowest passing): clean pass; fault 7/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P1 | HIP Stream-K two-tile (`hip-streamk-two-tile-m256-n256-k256`) | 🟩 high (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P2 | rocBLAS SGEMM square-64 (`rocblas-sgemm-square-64`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Tensile | P0 | gfx950 LDS-positive BF16 GEMM (`tensile-gfx950-lds-positive`) | 🟨 max: clean pass; fault 0/8 (bar 6/8); below bar | 🟩 delay-zero: clean pass; fault 8/8 (bar 6/8) |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 1/8 (bar 6/8); below bar |
| PyTorch | P0 | `torch.topk` (`pytorch-torch-topk`) | 🟩 higher (lowest passing): clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| PyTorch | P1 | `torch.sort` (`pytorch-torch-sort`) | 🟩 higher (lowest passing): clean pass; fault 7/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟨 max: clean pass; fault 0/8 (bar 6/8); below bar | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| PyTorch | P2 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟨 Out of scope: numerical pass; traced global-atomic kernels have no applicable LDS/FLAT race coverage | 🟨 Out of scope: numerical pass; traced global-atomic kernels have no applicable LDS/FLAT race coverage |
| PyTorch | P2 | norm/softmax (`pytorch-norm-softmax`) | 🟨 max: clean pass; fault 0/8 (bar 6/8); below bar | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |

Evidence is retained locally under
`/home/benjacob/consan-validation-gfx950-20260924/` (initial clean, inventories,
reviewed ISA, eleven single fault trials, source pins and `summary.json`) and
`/home/benjacob/consan-cdna4-qualification-20260924/` (prospective eight-trial
specs, matching controls, per-preset outcomes, retained objects and fixes).
The initial single trials do not satisfy the eight-trial qualification bar.

The initial sort stride-1 retirement trial detected a race despite its prior
miss expectation; it remains a rejected expectation contract and motivates
new prospective calibration, without rewriting its evidence. Tensile's initial
wrong-address fault was detected by SuperCollider and missed by Default, with
numerical failure in both modes. Its old duration floor incorrectly rejected
microsecond functional runs; the fix retains positive device timing and all
numerical/coverage checks and has passed physical baseline, Default and SuperCollider revalidation.
