# ConSan `gfx950` benchmark status

For each mode, **Startup** is the total latency through the first synchronized
run and its selected evidence checkpoints, including instrumentation, loading,
binding, and warm-up; **Run** is the absolute second-run latency followed
by its ratio to the matching uninstrumented second run.

| Workload | Uninstrumented Startup | Uninstrumented Run | Default Mode Startup | Default Mode Run | Default Mode (high) Startup | Default Mode (high) Run | SuperCollider Startup | SuperCollider Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | 3.12 s | 0.0028 s (1×) | 234 s | 0.00573 s (2.05×) | 235 s | 0.0737 s (26.3×) | 167 s | 0.00344 s (1.23×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/pytorch-dense-prefill--native-validation.log>) |
| PyTorch synthetic dense decode (one continuous-batch tick) | 54.4 s | 0.00336 s (1×) | 324 s | 0.00462 s (1.37×) | 321 s | 17.9 s (5,330×) | 258 s | 0.00364 s (1.08×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260918/pytorch-synthetic-decode/pytorch-synthetic-decode--native-validation.log>) |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending | [native-reference-1: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260918/pytorch-top1-moe-prefill/pytorch-top1-moe-prefill--native-reference-1.log>) |
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

Only the three incomplete rows are being rerun: synthetic decode, synthetic
MoE prefill, and TokenSpeed BF16 MoE. The previous artifact directory is not
present on this host; fresh native references will therefore accompany each
row so its ratios use matching inputs and software. Completed rows above are
preserved. Rebuilding ConSan alone does not invalidate them for this run, per
the accepted campaign policy.

Preparation complete: PyTorch matches the existing TheRock
`10.2.0a20260918` venv; TokenSpeed/Aorta use the original pinned sources.
The selected rows run serially, with cell transitions committed below.
New artifacts live under `/home/benjacob/work/consan-benchmark-gfx950-20260918/`.
