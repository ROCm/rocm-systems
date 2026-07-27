// Compiles the real paths.cc with all real RCCL headers.
// Stubs/ directory intercepts <hip/hip_runtime.h>, <cuda.h>, <hsa/hsa.h>.
#include "rccl_host_preamble.h"

#include "paths.cc"

// Expose static functions for testing.
ncclResult_t test_ncclTopoSetPaths(struct ncclTopoNode* baseNode,
                                   struct ncclTopoSystem* system) {
  return ncclTopoSetPaths(baseNode, system);
}
