// Minimal no-op kernel for LTTng curated-args coverage test fixtures.
// Compiled to a .hsaco code object that the coverage harness loads via
// hipModuleLoadData, then exercises hipModuleGetFunction, hipModuleLaunchKernel,
// hipModuleUnload against.
//
// The kernel body is intentionally empty — we don't care if it does any work,
// only that the module load/lookup/launch/unload chain succeeds end-to-end so
// the curated _args events fire.

#include <hip/hip_runtime.h>

extern "C" __global__ void noop_kernel() {
    // intentionally empty
}
