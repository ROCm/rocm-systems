# ConSan `gfx950` benchmark status

For each mode, **Startup** is the total latency through the first synchronized
run and its selected evidence checkpoints, including instrumentation, loading,
binding, and warm-up; **Run** is the absolute second-run latency followed
by its ratio to the matching uninstrumented second run.

| Workload | Uninstrumented Startup | Uninstrumented Run | SuperCollider Startup | SuperCollider Run | RecordReplay Startup | RecordReplay Run | Sampled Startup | Sampled Run | InlineShadow Startup | InlineShadow Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| PyTorch synthetic dense decode (one continuous-batch tick) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| Gluon verified shared-memory round trip (1024 elements) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Gluon BF16 GEMM medium-M (128×4096×4096) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | [kernel-inventory: accepted](</home/benjacob/consan-gfx950-native-admission/tokenspeed-bf16-gemm-mediumm--kernel-inventory.log>) |
| TokenSpeed Gluon BF16 GEMM large-M (4096×4096×4096) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | [kernel-inventory: accepted](</home/benjacob/consan-gfx950-native-admission/tokenspeed-bf16-gemm-largem--kernel-inventory.log>) |
| TokenSpeed Gluon attention prefill | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | [kernel-inventory: accepted](</home/benjacob/consan-gfx950-native-admission/tokenspeed-attention-prefill--kernel-inventory.log>) |
| TokenSpeed Gluon attention decode | pending | pending | pending | pending | pending | pending | pending | pending | fix-check passed; benchmark pending | fix-check passed; benchmark pending | [bugcheck-inline-shadow: fix-check passed; benchmark pending](</home/benjacob/consan-gfx950-attention-fix-check/tokenspeed-attention-decode--bugcheck-inline-shadow.log>) |
| TokenSpeed Qwen3-0.6B prefill | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Qwen3-0.6B real cached decode | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| TokenSpeed Triton FP8 block-scaled GEMM | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | [kernel-inventory: accepted](</home/benjacob/consan-gfx950-native-admission/tokenspeed-fp8-blockscale-gemm--kernel-inventory.log>) |
| TokenSpeed Gluon BF16 MoE | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | [kernel-inventory: accepted](</home/benjacob/consan-gfx950-native-admission/tokenspeed-bf16-moe--kernel-inventory.log>) |
