# Vulkan compute through RADV

Rocjitsu runs native compute shaders emitted by Mesa RADV for its RDNA3 and
RDNA4 models. The Vulkan loader, RADV, ACO and libdrm run normally; the Linux
interposer supplies the DRM device and sends PM4 submissions through the
existing command processor and wavefront execution path. Supported commands
include direct/indirect dispatch, shader registers, DMA copy/fill, memory waits
and timestamps on compute and graphics-capable queues.

## Running a workload

Build Rocjitsu with the shared interposer and use the same executable, shader
inputs, RADV and libdrm for simulator and hardware runs. From this directory's
parent, a typical Linux invocation is:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
LD_LIBRARY_PATH=/lib/x86_64-linux-gnu DRI_PRIME= \
/path/to/build/tools/rocjitsu/rocjitsu \
  --config configs/gfx1100_w7900.json -- /path/to/compute-workload
```

Use `gfx1201_r9700.json` for RDNA4. For hardware, omit the Rocjitsu launcher and
select the corresponding physical device with `DRI_PRIME`. Run IREE workloads
from a build with the Vulkan runtime driver and SPIR-V compiler backend enabled.
Compare individual test results as well as suite results: an additional skipped
case does not establish parity.

## Scope and limitations

Matched Vulkan workloads have been validated on W7900 (`gfx1100`) and R9700
(`gfx1201`). Full workload parity also requires the companion instruction and
buffer fixes.

Launch regressions cover RDNA1/2/3/3.5/4; RDNA1/2 CTS coverage is partial and
RDNA3.5 has no end-to-end RADV qualification.

Texel-buffer loads, stores and atomics use the buffer instruction path, including
packed format conversion. Graphics rendering, display, image load/store, sampling,
SDMA command streams and general Vulkan conformance remain outside this path.

Timestamps use a synthetic 1 GHz host monotonic clock; durations
measure simulator execution on the host.

Compute queues persist until VM teardown, and DRM files share the process GPU
address space. Submitted mappings must remain unchanged until completion;
submission-owned BO references preserve backing after handle closure. Device
teardown cancellation and overlapping dispatches sharing a scratch BO need
further work.
