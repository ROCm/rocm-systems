# ConSan `gfx1201` benchmark status

Last measured: 2026-09-11T06:30:34+00:00.

## Instrumentation overhead

Each cell is **audit-disabled instrumented latency / native latency**. Site-audit cost is excluded and reported separately below.

| Workload | SuperCollider | Record/Replay | Sampled | Inline Shadow |
| --- | ---: | ---: | ---: | ---: |
| PyTorch synthetic dense prefill (32-token prompt) | 79.775× | 98.076× | 96.571× | 97.064× |
| PyTorch synthetic dense decode (one continuous-batch tick) | 73.370× | 90.431× | 89.838× | 90.125× |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | 180.841× | 221.779× | 217.415× | 216.327× |

## Absolute latency

Milliseconds in the workload's synchronized first-use operation. Native is the median of the fresh-process measurements bracketing the four-mode matrix.

| Workload | Native | SuperCollider | Record/Replay | Sampled | Inline Shadow |
| --- | ---: | ---: | ---: | ---: | ---: |
| pytorch-dense-prefill | 830.022 | 66214.609 | 81405.318 | 80156.232 | 80565.363 |
| pytorch-synthetic-decode | 906.583 | 66515.784 | 81983.121 | 81445.583 | 81705.320 |
| pytorch-top1-moe-prefill | 1401.998 | 253538.522 | 310933.882 | 304815.747 | 303289.678 |

## Site-audit overhead

This is the separately measured process-wall delta from enabling detailed site evidence. Negative deltas are retained rather than clipped.

| Workload | Mode | Audit off | Audit on | Delta | Warm-run delta |
| --- | --- | ---: | ---: | ---: | ---: |
| pytorch-dense-prefill | SuperCollider | 137829.0 ms | 137764.0 ms | -65.0 ms (-0.05%) | +99.2 ms (+0.08%) |
| pytorch-dense-prefill | Record/Replay | 178143.9 ms | 178062.8 ms | -81.1 ms (-0.05%) | -386.5 ms (-0.25%) |
| pytorch-dense-prefill | Sampled | 163521.8 ms | 163849.7 ms | +327.9 ms (+0.20%) | +37.0 ms (+0.02%) |
| pytorch-dense-prefill | Inline Shadow | 163717.5 ms | 164599.1 ms | +881.6 ms (+0.54%) | +868.5 ms (+0.56%) |
| pytorch-synthetic-decode | SuperCollider | 138915.5 ms | 138650.9 ms | -264.6 ms (-0.19%) | -229.5 ms (-0.18%) |
| pytorch-synthetic-decode | Record/Replay | 176566.4 ms | 176829.8 ms | +263.4 ms (+0.15%) | +616.7 ms (+0.39%) |
| pytorch-synthetic-decode | Sampled | 165302.6 ms | 164557.4 ms | -745.2 ms (-0.45%) | -567.2 ms (-0.36%) |
| pytorch-synthetic-decode | Inline Shadow | 165653.1 ms | 165658.3 ms | +5.1 ms (+0.00%) | +416.3 ms (+0.27%) |
| pytorch-top1-moe-prefill | SuperCollider | 261983.3 ms | 264342.5 ms | +2359.2 ms (+0.90%) | +2261.5 ms (+0.89%) |
| pytorch-top1-moe-prefill | Record/Replay | 342974.3 ms | 343739.0 ms | +764.8 ms (+0.22%) | +513.8 ms (+0.17%) |
| pytorch-top1-moe-prefill | Sampled | 313578.3 ms | 313309.6 ms | -268.7 ms (-0.09%) | -63.9 ms (-0.02%) |
| pytorch-top1-moe-prefill | Inline Shadow | 312396.4 ms | 313766.8 ms | +1370.3 ms (+0.44%) | +1532.3 ms (+0.50%) |

### Site-audit coverage

`Checked` counts every discovered site examined by the audit; `missed` is supported minus patched. Every row also passed the final static and dynamic completeness verdict.

| Workload | Mode | Objects | Selected | Instrumented | Checked | Unsupported | Missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pytorch-dense-prefill | SuperCollider | 5 | 1810 | 1810 | 1810 | 0 | 0 |
| pytorch-dense-prefill | Record/Replay | 5 | 2066 | 2066 | 2066 | 0 | 0 |
| pytorch-dense-prefill | Sampled | 5 | 2066 | 2066 | 2066 | 0 | 0 |
| pytorch-dense-prefill | Inline Shadow | 5 | 2066 | 2066 | 2066 | 0 | 0 |
| pytorch-synthetic-decode | SuperCollider | 6 | 1766 | 1766 | 1766 | 0 | 0 |
| pytorch-synthetic-decode | Record/Replay | 6 | 2044 | 2044 | 2044 | 0 | 0 |
| pytorch-synthetic-decode | Sampled | 6 | 2044 | 2044 | 2044 | 0 | 0 |
| pytorch-synthetic-decode | Inline Shadow | 6 | 2044 | 2044 | 2044 | 0 | 0 |
| pytorch-top1-moe-prefill | SuperCollider | 9 | 1987 | 1987 | 1987 | 0 | 0 |
| pytorch-top1-moe-prefill | Record/Replay | 9 | 2373 | 2373 | 2373 | 0 | 0 |
| pytorch-top1-moe-prefill | Sampled | 9 | 2373 | 2373 | 2373 | 0 | 0 |
| pytorch-top1-moe-prefill | Inline Shadow | 9 | 2373 | 2373 | 2373 | 0 | 0 |

## Workloads and memory

| Workload | Parameters | Peak allocated (largest mode) | Primary metric |
| --- | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | 33,170,432 | 150.0 MiB | `prefill_latency_ms` |
| PyTorch synthetic dense decode (one continuous-batch tick) | 33,170,432 | 140.3 MiB | `decode_latency_ms` |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | 70,927,360 | 218.9 MiB | `prefill_latency_ms` |

## Provenance and scope

- Target: `gfx1201`; device: `AMD Radeon RX 9070` (`gfx1201`).
- ROCm Systems commit: `4803bdb385814d9ec73399b1a0b5a99c5ca0f9a1`.
- Aorta commit: `da894f76e4c14cec03f428e7aed2aa45dfbf6a65`.
- Hook SHA-256: `a24d2647d0d165da1fa0fb2e385cf8a6377ec8d2eeb007c268642a799e62025a`.
- PyTorch: `2.14.0.dev20260720+rocm7.1`; HIP: `7.1.52802`.
- Complete matrix wall time: 5123.6 seconds.
- TokenSpeed is not selected on gfx1201 because its pinned Aorta container rejects this target. Portable Gluon and explicitly pinned hipBLASLt/Tensile end-to-end cells remain future corpus additions.
- These synthetic Aorta models provide PyTorch prefill, synthetic decode, and top-1 MoE coverage; they are not real Qwen serving workloads.
- Each workload first inventories its native dispatches. Every exact observed kernel entry forms the ConSan allowlist; colocated but undispatched library kernels are outside the measurement policy.

The complete machine-readable samples and logs are retained in the artifact directory `/tmp/consan-benchmark-gfx1201-final`. See [BENCHMARK.md](BENCHMARK.md) for the measurement and admission contract.
