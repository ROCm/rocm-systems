# ConSan `gfx1201` benchmark status

For each mode, **Startup** is the total latency through the first synchronized
run and its selected evidence checkpoints, including instrumentation, loading,
binding, and warm-up; **Run** is the absolute second-run latency followed
by its ratio to the matching uninstrumented second run.

| Workload | Uninstrumented Startup | Uninstrumented Run | SuperCollider Startup | SuperCollider Run | ConSan Startup | ConSan Run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch synthetic dense prefill (32-token prompt) | 0.937 s | 0.00209 s (1×) | 132 s | 0.00332 s (1.59×) | 163 s | 0.0092 s (4.41×) |
| PyTorch synthetic dense decode (one continuous-batch tick) | 1.04 s | 0.00193 s (1×) | 134 s | 0.00495 s (2.56×) | 159 s | 0.00748 s (3.87×) |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | 1.23 s | 0.00745 s (1×) | 340 s | 0.0124 s (1.66×) | 412 s | 0.0215 s (2.89×) |
| Gluon verified shared-memory round trip (1024 elements) | 0.414 s | 0.000113 s (1×) | 0.404 s | 0.000113 s (0.999×) | 0.435 s | 0.000168 s (1.49×) |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | 0.0000165 s | 0.0000134 s (1×) | 0.521 s | 0.0000387 s (2.9×) | 0.785 s | 0.000127 s (9.52×) |

These measurements retain their campaign provenance; they do not constitute a
performance measurement of the current checkout.
