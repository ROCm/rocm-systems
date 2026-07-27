// Compiles NullParentTests.cpp with real RCCL topo types.
// Stubs/ directory intercepts <hip/hip_runtime.h>, <cuda.h>, <hsa/hsa.h>.
#include "rccl_host_preamble.h"

#include "topo.h"

// Wrapper function declarations
ncclResult_t test_ncclTopoSetPaths(struct ncclTopoNode* baseNode,
                                   struct ncclTopoSystem* system);
int test_gpuPciBw(struct ncclTopoNode* gpu);

#include "NullParentTests.cpp"
