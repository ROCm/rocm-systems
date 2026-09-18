# ConSan `gfx950` benchmark status

For each mode, **Startup** is the total latency through the first synchronized
run and its selected evidence checkpoints, including instrumentation, loading,
binding, and warm-up; **Run** is the absolute second-run latency followed
by its ratio to the matching uninstrumented second run.

| Workload | Uninstrumented Startup | Uninstrumented Run | Default Mode Startup | Default Mode Run | Default Mode (high) Startup | Default Mode (high) Run | SuperCollider Startup | SuperCollider Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | 3.14 s | 0.00279 s (1×) | 235 s | 0.00677 s (2.43×) | 235 s | 0.0752 s (27×) | 167 s | 0.00332 s (1.19×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/pytorch-dense-prefill--native-validation.log>) |
| PyTorch synthetic dense decode (one continuous-batch tick) | 2.76 s | 0.00266 s (1×) | 249 s | 0.00397 s (1.49×) | 250 s | 0.0499 s (18.8×) | 188 s | 0.00321 s (1.21×) | [native-validation: accepted](</home/benjacob/work/consan-benchmark-gfx950-20260917/pytorch-synthetic-decode--native-validation.log>) |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | 3.37 s | 0.00787 s (1×) | running | running | pending | pending | pending | pending | [default--audit-on: running](</home/benjacob/work/consan-benchmark-gfx950-20260917/pytorch-top1-moe-prefill--default--audit-on.log>) |
| Gluon verified shared-memory round trip (1024 elements) | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon BF16 GEMM medium-M (128×4096×4096) | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon BF16 GEMM large-M (4096×4096×4096) | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon attention prefill | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon attention decode | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Qwen3-0.6B prefill | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Qwen3-0.6B real cached decode | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Triton FP8 block-scaled GEMM | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon BF16 MoE | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
