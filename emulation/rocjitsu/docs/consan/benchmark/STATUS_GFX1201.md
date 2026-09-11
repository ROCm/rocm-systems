# ConSan `gfx1201` benchmark status

For each mode, **startup** is its incremental one-off cold-run cost and
**runtime** is steady-state instrumented latency / native latency.

| Workload | SuperCollider startup | SuperCollider runtime | RecordReplay startup | RecordReplay runtime | Sampled startup | Sampled runtime | InlineShadow startup | InlineShadow runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch synthetic dense prefill (32-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending |
| PyTorch synthetic dense decode (one continuous-batch tick) | pending | pending | pending | pending | pending | pending | pending | pending |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending |
