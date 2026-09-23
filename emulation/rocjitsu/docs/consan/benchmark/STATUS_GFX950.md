# ConSan `gfx950` benchmark status

Historical evidence: this ledger was not requalified during the September 2026
documentation audit. Existing results describe their recorded runs, not the
current branch. Rerun the linked procedure after relevant changes.

For each mode, **Startup** is the total latency through the first synchronized
run and its selected evidence checkpoints, including instrumentation, loading,
binding, and warm-up; **Run** is the absolute second-run latency followed
by its ratio to the matching uninstrumented second run.

| Workload | Uninstrumented Startup | Uninstrumented Run | Default Mode Startup | Default Mode Run | Default Mode (high) Startup | Default Mode (high) Run | SuperCollider Startup | SuperCollider Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | 3.12 s | 0.0028 s (1×) | 234 s | 0.00573 s (2.05×) | 235 s | 0.0737 s (26.3×) | 167 s | 0.00344 s (1.23×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/pytorch-dense-prefill--native-validation.log>) |
| PyTorch synthetic dense decode (one continuous-batch tick) | 54.4 s | 0.00336 s (1×) | 324 s | 0.00462 s (1.37×) | 321 s | 17.9 s (5,330×) | 258 s | 0.00364 s (1.08×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260918/pytorch-synthetic-decode/pytorch-synthetic-decode--native-validation.log>) |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | 92.7 s | 3.25 s (1×) | 566 s | 3.57 s (1.1×) | 612 s | 43.2 s (13.3×) | 460 s | 2.95 s (0.906×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260918/pytorch-top1-moe-prefill/pytorch-top1-moe-prefill--native-validation.log>) |
| Gluon verified shared-memory round trip (1024 elements) | 0.471 s | 0.0000865 s (1×) | 0.476 s | 0.000102 s (1.17×) | 0.477 s | 0.000106 s (1.23×) | 0.477 s | 0.000104 s (1.2×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/gluon-shared-roundtrip--native-validation.log>) |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | 0.000011 s | 0.00000986 s (1×) | 5.61 s | 0.000399 s (40.5×) | 5.59 s | 0.00693 s (703×) | 3.47 s | 0.000016 s (1.63×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/hipblaslt-tensile-gemm--native-validation.log>) |
| TokenSpeed Gluon BF16 GEMM medium-M (128×4096×4096) | 0.095 s | 0.000527 s (1×) | 26.5 s | 0.00464 s (8.8×) | 27.2 s | 0.0862 s (163×) | 23 s | 0.000829 s (1.57×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/tokenspeed-bf16-gemm-mediumm--native-validation.log>) |
| TokenSpeed Gluon BF16 GEMM large-M (4096×4096×4096) | 0.092 s | 0.000359 s (1×) | 26.6 s | 0.0212 s (59×) | 27.1 s | 0.516 s (1,440×) | 22.5 s | 0.00148 s (4.14×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/tokenspeed-bf16-gemm-largem--native-validation.log>) |
| TokenSpeed Gluon attention prefill | 0.149 s | 0.000291 s (1×) | 26.7 s | 0.00129 s (4.43×) | 27.3 s | 0.0277 s (95.1×) | 22.5 s | 0.000494 s (1.69×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/tokenspeed-attention-prefill--native-validation.log>) |
| TokenSpeed Gluon attention decode | 0.137 s | 0.000323 s (1×) | 26.5 s | 0.000334 s (1.03×) | 26.6 s | 0.000396 s (1.23×) | 22.6 s | 0.000333 s (1.03×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/tokenspeed-attention-decode--native-validation.log>) |
| TokenSpeed Qwen3-0.6B prefill | 2.72 s | 0.0182 s (1×) | 265 s | 0.044 s (2.41×) | 266 s | 0.703 s (38.6×) | 192 s | 0.0196 s (1.07×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/tokenspeed-qwen-prefill--native-validation.log>) |
| TokenSpeed Qwen3-0.6B real cached decode | 0.0265 s | 0.0179 s (1×) | 267 s | 0.043 s (2.4×) | 266 s | 0.621 s (34.7×) | 187 s | 0.0191 s (1.06×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/tokenspeed-qwen-decode--native-validation.log>) |
| TokenSpeed Triton FP8 block-scaled GEMM | 0.0785 s | 0.000356 s (1×) | 26.2 s | 0.000373 s (1.05×) | 26.3 s | 0.000358 s (1.01×) | 22.3 s | 0.000395 s (1.11×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/tokenspeed-fp8-blockscale-gemm--native-validation.log>) |
| TokenSpeed Gluon BF16 MoE | 3.3 s | 3.22 s (1×) | N/A (no applicable sites) | N/A (no applicable sites) | N/A (no applicable sites) | N/A (no applicable sites) | N/A (no applicable sites) | N/A (no applicable sites) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260918/tokenspeed-bf16-moe/tokenspeed-bf16-moe--native-validation.log>) |

## Completion run (September 18, 2026)

Only the three previously incomplete rows were rerun: synthetic decode,
synthetic MoE prefill, and TokenSpeed BF16 MoE. All their cells are now complete;
all ten previously complete rows are preserved. The previous artifact directory
is absent on this host, so each rerun uses fresh, matching native references.
The venv uses TheRock `10.2.0a20260918` and matching PyTorch, with the original
pinned TokenSpeed/Aorta sources. The ConSan hook binary was unchanged.

TokenSpeed BF16 MoE is **N/A in all three modes**: every selected kernel loaded
and dispatched, all static inventories completed with zero applicable sites,
and all numerical checks passed. Its small shape selects register-only wave
GEMVs. These outcomes are not sanitizer performance measurements.

Final native Run drift was **−1.02%** for synthetic decode, **+9.46%** for
synthetic MoE, and **−0.72%** for TokenSpeed MoE. Synthetic MoE's initial native
Run samples were **2.95–3.55 s**, with a final sample of **3.56 s**. Its Default
Mode and SuperCollider ratios fall within that variation; the SuperCollider
ratio below 1× is **not evidence of a speedup**. High mode's measured costs
(17.9 s for decode and 43.2 s for synthetic MoE) remain reported as observed.

The [completion run notes](GFX950.md#september-18-completion-run) explain the
classification fix, provenance, and timing limitations. New artifacts are under
`/home/benjacob/work/consan-benchmark-gfx950-20260918/`. The
[completion audit](</home/benjacob/work/consan-benchmark-prerequisites-20260918/completion-audit.json>)
verified all 21 new checkpoints against their logs, recomputed the table values,
and confirmed that the completed historical rows and hook hash were unchanged.
