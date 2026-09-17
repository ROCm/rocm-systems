# ConSan `gfx950` benchmark status

For each mode, **Startup** is the total latency through the first synchronized
run and its selected evidence checkpoints, including instrumentation, loading,
binding, and warm-up; **Run** is the absolute second-run latency followed
by its ratio to the matching uninstrumented second run.

| Workload | Uninstrumented Startup | Uninstrumented Run | SuperCollider Startup | SuperCollider Run | ConSan Startup | ConSan Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | 4.3 s | 0.00351 s (1×) | 185 s | 0.00401 s (1.14×) | 226 s | 0.00774 s (2.2×) | [native-validation: resumed](</home/benjacob/consan-gfx950-benchmark-full/pytorch-dense-prefill--native-validation.log>) |
| PyTorch synthetic dense decode (one continuous-batch tick) | 3.73 s | 0.00361 s (1×) | 190 s | 0.00346 s (0.958×) | 252 s | 0.0077 s (2.13×) | [native-validation: resumed](</home/benjacob/consan-gfx950-benchmark-full/pytorch-synthetic-decode--native-validation.log>) |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | 4.64 s | 0.0112 s (1×) | 326 s | 0.0137 s (1.22×) | 451 s | 0.021 s (1.87×) | [native-validation: resumed](</home/benjacob/consan-gfx950-benchmark-full/pytorch-top1-moe-prefill--native-validation.log>) |
| Gluon verified shared-memory round trip (1024 elements) | 0.608 s | 0.0000899 s (1×) | 0.612 s | 0.000127 s (1.41×) | 0.617 s | 0.000109 s (1.21×) | [native-validation: resumed](</home/benjacob/consan-gfx950-benchmark-full/gluon-shared-roundtrip--native-validation.log>) |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | 0.0000133 s | 0.0000115 s (1×) | 4.8 s | 0.0000165 s (1.44×) | 7.8 s | 0.000382 s (33.4×) | [native-validation: resumed](</home/benjacob/consan-gfx950-benchmark-full/hipblaslt-tensile-gemm--native-validation.log>) |
| TokenSpeed Gluon BF16 GEMM medium-M (128×4096×4096) | 0.124 s | 0.000628 s (1×) | 1.28 s | 0.00122 s (1.94×) | 5.38 s | 0.00536 s (8.54×) | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-bf16-gemm-mediumm--native-validation.log>) |
| TokenSpeed Gluon BF16 GEMM large-M (4096×4096×4096) | 0.121 s | 0.000649 s (1×) | 1.32 s | 0.00177 s (2.73×) | 5.39 s | 0.0213 s (32.8×) | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-bf16-gemm-largem--native-validation.log>) |
| TokenSpeed Gluon attention prefill | 0.126 s | 0.000624 s (1×) | 1.37 s | 0.000818 s (1.31×) | 5.4 s | 0.00282 s (4.52×) | [native-validation: resumed](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-attention-prefill--native-validation.log>) |
| TokenSpeed Gluon attention decode | 0.128 s | 0.000664 s (1×) | 1.33 s | 0.000687 s (1.04×) | 5.35 s | 0.000687 s (1.04×) | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-attention-decode--native-validation.log>) |
| TokenSpeed Qwen3-0.6B prefill | 3.77 s | 0.0255 s (1×) | 173 s | 0.0275 s (1.08×) | 232 s | 0.0463 s (1.81×) | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-qwen-prefill--native-validation.log>) |
| TokenSpeed Qwen3-0.6B real cached decode | 0.0384 s | 0.0255 s (1×) | 171 s | 0.0267 s (1.05×) | 227 s | 0.0417 s (1.64×) | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-qwen-decode--native-validation.log>) |
| TokenSpeed Triton FP8 block-scaled GEMM | 0.105 s | 0.000722 s (1×) | 1.28 s | 0.00073 s (1.01×) | 5.32 s | 0.000725 s (1×) | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-fp8-blockscale-gemm--native-validation.log>) |
| TokenSpeed Gluon BF16 MoE | 0.109 s | 0.000676 s (1×) | not applicable (no admitted sites) | not applicable (no admitted sites) | not applicable (no admitted sites) | not applicable (no admitted sites) | [native-validation: accepted](</home/benjacob/consan-gfx950-benchmark-full/tokenspeed-bf16-moe--native-validation.log>) |

The ConSan and SuperCollider measurements cover 13 workloads: 24 passing
mode cells and two inapplicable cells. The BF16 MoE kernels have no admitted
instrumentation sites; no instrumentation overhead is claimed for those cells.
Passing cells require numerical correctness and complete static instrumentation;
dynamic-analysis limitations remain recorded in the artifacts. Measurements
are tied to the source and hook provenance in those artifacts, not automatically
to the current checkout.

See [campaign details](GFX950.md) for the environment, reproduction command,
retention policy, and evidence locations.
