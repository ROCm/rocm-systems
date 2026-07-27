// Compiles the real paths.cc (hipified) with hipcc --offload-host-only.
// No stubs needed — hipcc handles HIP/CUDA headers natively.
#include "paths.cc"

ncclResult_t test_ncclTopoSetPaths(struct ncclTopoNode* baseNode,
                                   struct ncclTopoSystem* system) {
  return ncclTopoSetPaths(baseNode, system);
}
