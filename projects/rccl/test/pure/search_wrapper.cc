// Compiles the real search.cc with all real RCCL headers.
// Stubs/ directory intercepts <hip/hip_runtime.h>, <cuda.h>, <hsa/hsa.h>.
#include "rccl_host_preamble.h"

#include "search.cc"

// Expose static functions for testing.
int test_gpuPciBw(struct ncclTopoNode* gpu) {
  return gpuPciBw(gpu);
}
