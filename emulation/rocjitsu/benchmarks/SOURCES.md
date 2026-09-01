# Benchmark source attribution

The rocjitsu adapters are original fixed workloads built on public HIP, Triton,
and hipBLASLt interfaces. They do not load code objects or call `rj_vm`
directly. Package versions are pinned in `requirements.txt`; generated kernels,
caches, and HSCO files are build artifacts and must not be committed.

## Triton workloads

`workloads/triton_workloads.py` uses the public language and launch interfaces
from <https://github.com/ROCm/triton>. The workloads use fixed launch
configurations and do not autotune while measuring. The aligned and boundary
variants are local coverage rather than copies of an upstream parameter sweep.

## hipBLASLt/TensileLite workloads

`src/hipblaslt_workloads.cpp` expresses three fixed problems through the public
hipBLASLt API maintained in
<https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt>. It uses
hipBLASLt from the installed `rocm-sdk-libraries` package and the target-specific
TensileLite assets from `rocm-sdk-device-gfx950` and
`rocm-sdk-device-gfx1250`. No rocm-libraries checkout, device-library generation,
source build, tuning loop, or CPU verification is part of the benchmark
workflow.
