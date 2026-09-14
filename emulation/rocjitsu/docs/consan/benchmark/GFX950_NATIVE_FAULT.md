# gfx950 native GPU fault investigation

Fresh processes now report GPU illegal memory accesses with ConSan disabled.
This prevents attributing the attention-prefill RecordReplay failure to ConSan.
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

Other processes still hold GPU memory: PIDs 2509904, 2510869, and 2512712.
No reset or interruption of these jobs has been performed. Approval has been
requested before any recovery action that would interrupt those jobs.
After recovery, require a passing native HIP smoke and attention control,
then resume remaining benchmark cells with `--timeout 600 --resume`.
Do not discard completed cells or retry the capped large-M InlineShadow cell.
