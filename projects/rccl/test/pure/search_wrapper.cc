// Compiles the real search.cc (hipified) with hipcc --offload-host-only.
// No stubs needed — hipcc handles HIP/CUDA headers natively.
#include "search.cc"

int test_gpuPciBw(struct ncclTopoNode* gpu) {
  return gpuPciBw(gpu);
}
