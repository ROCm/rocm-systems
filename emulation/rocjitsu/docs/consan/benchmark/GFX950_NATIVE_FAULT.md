# gfx950 native GPU fault investigation

Fresh processes reported GPU illegal memory accesses with ConSan disabled.
A GPU reset restored the standalone native HIP smoke test. The root cause
of the GPU state failure is not yet established.
Completed benchmark measurements and their original provenance are retained.

Reproductions use `/home/benjacob/consan-prerequisites/env.sh`, the venv TheRock
runtime, and physical MI350X at PCI `0000:83:00.0`. Each process is capped at
600 seconds; diagnostics that already reported faults were stopped earlier.

- The uninstrumented attention-prefill adapter fails both normally and with
  serialized HIP launches.
- A standalone PyTorch `arange` and device-to-host copy also faults.
- The existing `hip-smoke-normal` binary fails in `lds_exchange` without
  ConSan, PyTorch, or TokenSpeed loaded.
- Process-local `HSA_ENABLE_SDMA=0` and `HIP_FORCE_DEV_KERNARG=0` controls
  also fail. No environment workaround has been adopted for measurements.
- Precise-memory rocgdb reporting captures a fault in a runtime copy kernel.
  An earlier imprecise report stopped in attention; that location alone does
  not establish the cause.

Evidence: `/home/benjacob/consan-prerequisites/evidence/native-gpu-fault/`.
Additional logs and dumped code objects are in
`/home/benjacob/consan-gfx950-attention-prefill-repro/`.

With explicit user approval, PIDs 2509904, 2510869, and 2512712 were stopped.
AMD SMI confirmed no remaining GPU processes. Native HIP still failed.
The authorized `amd-smi reset --gpureset --gpu 0` then succeeded, and the
unchanged native HIP binary passed all 65536 CPU-oracle comparisons.
The unchanged native attention control also passed after reset. Remaining
benchmark cells resumed with `--timeout 600 --resume`.
Do not discard completed cells or retry the capped large-M InlineShadow cell.

The first post-reset full RecordReplay attempt still faults, while an immediate
standalone native HIP check passes. Thus GPU recovery did not resolve the
RecordReplay issue; instrumentation isolation continues on the recovered GPU.
