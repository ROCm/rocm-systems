# ConSan `gfx1201` benchmark status

For each mode, **Startup** is the one-off instrumentation/load cost;
**Run1** and **Run2** are matching-order instrumented/native overhead ratios.

| Workload | SuperCollider Startup | SuperCollider Run1 | SuperCollider Run2 | RecordReplay Startup | RecordReplay Run1 | RecordReplay Run2 | Sampled Startup | Sampled Run1 | Sampled Run2 | InlineShadow Startup | InlineShadow Run1 | InlineShadow Run2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch synthetic dense prefill (32-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| PyTorch synthetic dense decode (one continuous-batch tick) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| Gluon verified shared-memory round trip (1024 elements) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
