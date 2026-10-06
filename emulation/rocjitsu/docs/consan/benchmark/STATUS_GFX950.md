# ConSan `gfx950` benchmark status

Rerun on **October 5, 2026**, on a physical MI350X with TheRock
`10.2.0a20261005`, HIP `7.17.26392`, PyTorch
`2.15.0a0+rocm10.2.0a20261005`, and Python `3.12.15`. The measured
RocJITsu revision is `c88681ad6a` from `shared/rocjitsu/sanitizers`,
preserved across the campaign interruption; hook SHA256 is
`903301e8b07926c8a60344b78fdce16c6c87fbb2006f675573c8a21ea9a80d19`.
All thirteen rows have fresh profiler discovery, two native reference processes,
three instrumented modes, and a final native numerical/drift check.
Only the venv's TheRock ROCm was used.

For each mode, **Startup** sums instrumentation and the measured first run;
**Run** is the absolute second-run latency followed by its ratio to the
matching uninstrumented second run. PyTorch/Gluon use synchronized host
timing. hipBLASLt uses GPU event timing, so its Startup excludes host client
setup and is not an end-to-end cold-start measurement.

| Workload | Uninstrumented Startup | Uninstrumented Run | Default Mode Startup | Default Mode Run | Default Mode (high) Startup | Default Mode (high) Run | SuperCollider Startup | SuperCollider Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | 4.47 s | 0.0036 s (1×) | 26.8 s | 0.0107 s (2.98×) | 27.3 s | 0.101 s (28.2×) | 34.9 s | 0.00371 s (1.03×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-without-hipblaslt/pytorch-dense-prefill--native-validation.log>) |
| PyTorch synthetic dense decode (one continuous-batch tick) | 3.84 s | 0.00342 s (1×) | 21.7 s | 0.0053 s (1.55×) | 21.5 s | 0.0586 s (17.1×) | 32.4 s | 0.00379 s (1.11×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-without-hipblaslt/pytorch-synthetic-decode--native-validation.log>) |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | 4.79 s | 0.0112 s (1×) | 42.1 s | 0.0247 s (2.2×) | 42 s | 0.194 s (17.3×) | 61.9 s | 0.0125 s (1.12×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-without-hipblaslt/pytorch-top1-moe-prefill--native-validation.log>) |
| Gluon verified shared-memory round trip (1024 elements) | 0.614 s | 0.000564 s (1×) | 0.618 s | 0.000487 s (0.864×) | 0.619 s | 0.000508 s (0.9×) | 0.618 s | 0.000592 s (1.05×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-without-hipblaslt/gluon-shared-roundtrip--native-validation.log>) |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | 0.0000134 s | 0.0000108 s (1×) | 4.65 s | 0.000426 s (39.5×) | 4.69 s | 0.00695 s (644×) | 3.33 s | 0.0000366 s (3.39×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-final-five/hipblaslt-tensile-gemm--native-validation.log>) |
| TokenSpeed Gluon BF16 GEMM medium-M (128×4096×4096) | 0.146 s | 0.000647 s (1×) | 1.37 s | 0.00495 s (7.65×) | 1.46 s | 0.0833 s (129×) | 3.39 s | 0.00107 s (1.65×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-without-hipblaslt/tokenspeed-bf16-gemm-mediumm--native-validation.log>) |
| TokenSpeed Gluon BF16 GEMM large-M (4096×4096×4096) | 0.143 s | 0.000763 s (1×) | 1.4 s | 0.0226 s (29.6×) | 1.86 s | 0.487 s (638×) | 3.38 s | 0.00182 s (2.39×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-without-hipblaslt/tokenspeed-bf16-gemm-largem--native-validation.log>) |
| TokenSpeed Gluon attention prefill | 0.152 s | 0.000643 s (1×) | 1.37 s | 0.00154 s (2.4×) | 1.39 s | 0.0254 s (39.5×) | 3.38 s | 0.000844 s (1.31×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-without-hipblaslt/tokenspeed-attention-prefill--native-validation.log>) |
| TokenSpeed Gluon attention decode | 0.15 s | 0.000659 s (1×) | 1.38 s | 0.000722 s (1.1×) | 1.39 s | 0.00114 s (1.73×) | 3.38 s | 0.000683 s (1.04×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-without-hipblaslt/tokenspeed-attention-decode--native-validation.log>) |
| TokenSpeed Qwen3-0.6B prefill | 3.2 s | 0.0235 s (1×) | 22.4 s | 0.0456 s (1.94×) | 23 s | 0.736 s (31.3×) | 33.2 s | 0.0257 s (1.09×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-final-five/tokenspeed-qwen-prefill--native-validation.log>) |
| TokenSpeed Qwen3-0.6B real cached decode | 0.0412 s | 0.0232 s (1×) | 19.1 s | 0.0445 s (1.92×) | 20 s | 0.733 s (31.7×) | 30 s | 0.0253 s (1.09×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-final-five/tokenspeed-qwen-decode--native-validation.log>) |
| TokenSpeed Triton FP8 block-scaled GEMM | 0.108 s | 0.000704 s (1×) | 1.35 s | 0.000729 s (1.04×) | 1.32 s | 0.000736 s (1.05×) | 3.31 s | 0.000736 s (1.05×) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-final-five/tokenspeed-fp8-blockscale-gemm--native-validation.log>) |
| TokenSpeed Gluon BF16 MoE | 0.109 s | 0.000721 s (1×) | N/A (no applicable sites) | N/A (no applicable sites) | N/A (no applicable sites) | N/A (no applicable sites) | N/A (no applicable sites) | N/A (no applicable sites) | [native-validation: accepted](</home/benjacob/consan-rerun-20261005/benchmark-final-five/tokenspeed-bf16-moe--native-validation.log>) |

Numerics and complete static coverage passed in all 36 applicable mode cells.
TokenSpeed BF16 MoE is **N/A in all three modes**: the selected kernels loaded
and dispatched, their complete inventories contain zero applicable sites, and
numerics passed. These cells provide no sanitizer overhead measurement.
See [global-memory scope](../SUPERCOLLIDER_GLOBAL_MEMORY.md).
Dynamic race reports and bounded retention remain in the artifacts; timing
admission does not establish the validation matrix's fault qualification.

Final native Run drift exceeded 5% for `pytorch-synthetic-decode` (+14.81%), `hipblaslt-tensile-gemm` (-18.01%). Their ratios retain that baseline-drift caveat. Small host-timed cells are single synchronized operations; ratios below 1× do not establish a speedup. All raw native samples and final checks are retained.

Aorta is `73409f79a56ac10ae5374b419e5737221daadc34`; TokenSpeed is
`a0cc3bb4d2b46e9995ae79a5ca1966aee00abc68`, with its Triton/Proton
`3.8.10.post20260906` and Transformers `5.12.0`. The hipBLASLt client uses
source `916d478bccb056ccc8da2709fc26312d1dee3b7e`, matching the installed
TheRock library, and links the SDK's runtime libraries.

[Combined evidence](</home/benjacob/consan-rerun-20261005/benchmark-summary.json>)
contains all thirteen rows and exact per-row provenance. The first eight rows
and final five were run as separate serialized batches, each with matching
fresh native controls. The [completion audit](</home/benjacob/consan-rerun-20261005/benchmark-completion-audit.json>)
verified all 91 checkpoints against their logs, reconstructed the profiler
allowlists, and recomputed every table value. Initial prerequisite failures were retained, corrected,
and excluded from the table. [Preparation and commands](</home/benjacob/consan-rerun-20261005/README.md>)
record the environment, native Tensile wrapper, SDK HIP-library path setting,
external-source patches, and package identities. Follow [BENCHMARK.md](BENCHMARK.md)
for the admission and timing contract; [GFX950.md](GFX950.md) preserves the older
September campaign and its historical values.
